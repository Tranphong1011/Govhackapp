import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime
import hashlib
import json
import io
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
import os
import random
from database import UserDB
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

st.set_page_config(
    page_title="Government Data Conversational AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("🔑 OPENAI_API_KEY not found in secrets.toml. Please add it to .streamlit/secrets.toml")
    st.stop()

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'data_sources' not in st.session_state:
    st.session_state.data_sources = {}
if 'audit_trail' not in st.session_state:
    st.session_state.audit_trail = []
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = []
if 'dataset_questions' not in st.session_state:
    st.session_state.dataset_questions = {}
if 'uploaded_files_hash' not in st.session_state:
    st.session_state.uploaded_files_hash = ""
if 'profile_reports' not in st.session_state:
    st.session_state.profile_reports = {}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

class DataProcessor:
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls']
    
    def load_file(self, uploaded_file):
        """Load and process uploaded file with multiple encoding support"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try multiple encodings for CSV files
                encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
                df = None
                
                for encoding in encodings_to_try:
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"✅ File loaded successfully with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        # If it's not an encoding issue, try next encoding
                        if 'codec' not in str(e).lower():
                            break
                        continue
                
                if df is None:
                    # Last resort: try with error handling
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')
                        st.warning("⚠️ File loaded with some characters ignored due to encoding issues")
                    except Exception as e:
                        return None, f"Unable to read CSV file with any encoding: {str(e)}"
                        
            elif file_extension in ['xlsx', 'xls']:
                try:
                    df = pd.read_excel(uploaded_file)
                except Exception as e:
                    return None, f"Error reading Excel file: {str(e)}"
            else:
                return None, f"Unsupported file format: {file_extension}"
            
            # Validate dataframe
            if df is None or df.empty:
                return None, "File appears to be empty or could not be read"
            
            metadata = {
                'filename': uploaded_file.name,
                'size': uploaded_file.size,
                'columns': list(df.columns),
                'rows': len(df),
                'upload_time': datetime.now().isoformat(),
                'dataframe': df
            }
            
            return df, metadata
            
        except Exception as e:
            return None, f"Error loading file: {str(e)}"
    
    def combine_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple datasets intelligently"""
        if not datasets:
            return pd.DataFrame()
        
        combined_df = pd.DataFrame()
        for name, df in datasets.items():
            df_copy = df.copy()
            df_copy['source_dataset'] = name
            if combined_df.empty:
                combined_df = df_copy
            else:
                combined_df = pd.concat([combined_df, df_copy], ignore_index=True, sort=False)
        
        return combined_df
    
    def get_files_hash(self, uploaded_files) -> str:
        """Generate hash from uploaded files to detect changes"""
        if not uploaded_files:
            return ""
        file_info = [(f.name, f.size) for f in uploaded_files]
        return hashlib.md5(str(sorted(file_info)).encode()).hexdigest()

class ProfileReportGenerator:
    def __init__(self):
        pass
    
    def generate_profile_report(self, df: pd.DataFrame, dataset_name: str, minimal: bool = True):
        """Generate ProfileReport for a dataset"""
        try:
            if len(df) > 1000:
                df_sample = df.sample(n=1000, random_state=42)
                st.warning(f"⚡ Using sample of 1000 rows from {len(df)} total rows for faster processing.")
            else:
                df_sample = df
                
            profile = ProfileReport(
                df_sample,
                title=f"{dataset_name} Profile Report",
                minimal=minimal,
                explorative=True
            )
            
            return profile
            
        except Exception as e:
            st.error(f"❌ Error generating profile report: {str(e)}")
            st.info("💡 This might be due to data type issues or memory constraints.")
            return None
    
    def display_profile_report(self, profile: ProfileReport, key: str):
        """Display ProfileReport in Streamlit"""
        if profile is None:
            st.error("No profile report to display")
            return
            
        try:
            st_profile_report(profile, navbar=True, key=key)
            
        except ImportError:
            st.warning("📦 Installing streamlit-ydata-profiling recommended for better display")
            try:
                profile_html = profile.to_html()
                st.components.v1.html(profile_html, height=800, scrolling=True)
            except Exception as e:
                st.error(f"❌ Cannot display profile report: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Display error: {str(e)}")

class DatasetConnectionAnalyzer:
    def __init__(self):
        pass
    
    def find_connections(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Find connections between datasets"""
        connections = {
            'common_columns': {},
            'similar_values': {},
            'potential_joins': {},
            'domain_relationships': {}
        }
        
        dataset_names = list(datasets.keys())
        
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                df1, df2 = datasets[name1], datasets[name2]
                common_cols = set(df1.columns).intersection(set(df2.columns))
                common_cols.discard('source_dataset')
                
                if common_cols:
                    connections['common_columns'][f"{name1}_{name2}"] = list(common_cols)
        
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                df1, df2 = datasets[name1], datasets[name2]
                
                id_cols_1 = [col for col in df1.columns if 'id' in col.lower() or 'code' in col.lower()]
                id_cols_2 = [col for col in df2.columns if 'id' in col.lower() or 'code' in col.lower()]
                
                if id_cols_1 and id_cols_2:
                    connections['potential_joins'][f"{name1}_{name2}"] = {
                        'dataset1_keys': id_cols_1,
                        'dataset2_keys': id_cols_2
                    }
        
        connections['domain_relationships'] = self._analyze_domain_relationships(datasets)
        
        return connections
    
    def _analyze_domain_relationships(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Analyze relationships between different data domains"""
        relationships = {}
        
        for name, df in datasets.items():
            columns_lower = [col.lower() for col in df.columns]
            
            if any(keyword in ' '.join(columns_lower) for keyword in ['budget', 'cost', 'expense', 'revenue']):
                domain = 'Finance'
            elif any(keyword in ' '.join(columns_lower) for keyword in ['employee', 'staff', 'leave', 'performance']):
                domain = 'HR'
            elif any(keyword in ' '.join(columns_lower) for keyword in ['process', 'efficiency', 'service', 'delivery']):
                domain = 'Operations'
            else:
                domain = 'General'
            
            relationships[name] = domain
        
        return relationships

class QuestionGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.connection_analyzer = DatasetConnectionAnalyzer()
    
    def generate_questions_for_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Generate questions for individual datasets and combined analysis"""
        all_questions = {}
        
        for name, df in datasets.items():
            try:
                questions = self._generate_single_dataset_questions(df, name)
                all_questions[name] = questions
            except Exception as e:
                st.warning(f"Could not generate questions for {name}: {str(e)}")
                all_questions[name] = self._get_fallback_questions()
        
        if len(datasets) > 1:
            try:
                cross_questions = self._generate_cross_dataset_questions(datasets)
                all_questions['cross_dataset'] = cross_questions
            except Exception as e:
                st.warning(f"Could not generate cross-dataset questions: {str(e)}")
                all_questions['cross_dataset'] = []
        
        return all_questions
    
    def select_top_questions(self, all_questions: Dict[str, List[str]], num_questions: int = 5) -> List[str]:
        """Select top questions from all available questions"""
        question_pool = []
        
        for dataset_name, questions in all_questions.items():
            for question in questions:
                question_pool.append({
                    'question': question,
                    'source': dataset_name,
                    'priority': self._calculate_question_priority(question, dataset_name)
                })
        
        question_pool = sorted(question_pool, key=lambda x: x['priority'], reverse=True)
        
        selected = []
        used_sources = set()
        
        for item in question_pool:
            if len(selected) >= num_questions:
                break
            if item['source'] not in used_sources or len(used_sources) == len(all_questions):
                selected.append(item['question'])
                used_sources.add(item['source'])
        
        remaining_questions = [item['question'] for item in question_pool if item['question'] not in selected]
        while len(selected) < num_questions and remaining_questions:
            selected.append(remaining_questions.pop(random.randint(0, len(remaining_questions)-1)))
        
        return selected[:num_questions]
    
    def _generate_single_dataset_questions(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        """Generate questions for a single dataset"""
        data_summary = self._analyze_data_structure(df)
        data_domain = self._detect_data_domain(data_summary)
        
        prompt = f"""
        Based on the following dataset "{dataset_name}" ({data_domain} related), generate exactly 3 specific questions that would help analyze this data:

        Dataset Information:
        - Name: {dataset_name}
        - Domain: {data_domain}
        - Total Records: {data_summary['total_rows']}
        - Columns: {', '.join(data_summary['columns'])}
        - Key Fields: {', '.join(data_summary['potential_amount_fields'] + data_summary['potential_id_fields'])}

        Sample Data: {data_summary['sample_data'][0] if data_summary['sample_data'] else 'No data'}

        Generate SHORT, concise questions (maximum 8-10 words each) focusing on key insights, trends, and outliers.
        Format: Return exactly 3 short questions, each on a new line, numbered 1-3.
        Examples: "Budget variance by department?", "Top spending categories?", "Performance trends?"
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate specific, actionable questions for government data analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        return self._parse_questions(response.choices[0].message.content)
    
    def _generate_cross_dataset_questions(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Generate questions that connect multiple datasets"""
        connections = self.connection_analyzer.find_connections(datasets)
        
        dataset_summary = {}
        for name, df in datasets.items():
            data_summary = self._analyze_data_structure(df)
            dataset_summary[name] = {
                'domain': self._detect_data_domain(data_summary),
                'key_columns': data_summary['potential_amount_fields'] + data_summary['potential_id_fields'],
                'total_records': len(df)
            }
        
        prompt = f"""
        Based on the following multiple government datasets, generate exactly 4 questions that analyze relationships and connections between the datasets:

        Datasets:
        {json.dumps(dataset_summary, indent=2)}
        
        Connections Found:
        - Common Columns: {connections['common_columns']}
        - Domain Relationships: {connections['domain_relationships']}
        - Potential Joins: {connections['potential_joins']}

        Generate questions that:
        - Compare metrics across datasets
        - Identify correlations between different data sources
        - Find insights that emerge only when datasets are combined
        - Highlight discrepancies or patterns across departments

        Generate SHORT, concise questions (maximum 8-10 words each) that analyze relationships between datasets.
        Format: Return exactly 4 short questions, each on a new line, numbered 1-4.
        Examples: "Budget vs performance correlation?", "Cross-department efficiency comparison?", "Resource allocation patterns?"
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate cross-dataset analysis questions for government data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.3
        )
        
        return self._parse_questions(response.choices[0].message.content)
    
    def _calculate_question_priority(self, question: str, source: str) -> float:
        """Calculate priority score for question selection"""
        priority = 1.0
        
        if source == 'cross_dataset':
            priority += 0.5
        
        high_priority_keywords = ['trend', 'outlier', 'compare', 'correlation', 'pattern', 'efficiency', 'budget']
        for keyword in high_priority_keywords:
            if keyword.lower() in question.lower():
                priority += 0.2
                break
        
        if any(char.isdigit() for char in question):
            priority += 0.1
        
        return priority
    
    def _analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the structure and content of the dataframe"""
        analysis = {
            'columns': list(df.columns),
            'total_rows': len(df),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'date_columns': list(df.select_dtypes(include=['datetime64']).columns),
            'sample_data': df.head(3).to_dict('records'),
            'missing_data': df.isnull().sum().to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns if df[col].dtype == 'object'}
        }
        
        analysis['potential_id_fields'] = [col for col in df.columns if 'id' in col.lower() or 'code' in col.lower()]
        analysis['potential_amount_fields'] = [col for col in df.columns if any(keyword in col.lower() for keyword in ['amount', 'cost', 'budget', 'salary', 'revenue', 'expense', 'price', 'value'])]
        analysis['potential_date_fields'] = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day'])]
        
        return analysis
    
    def _detect_data_domain(self, data_summary: Dict) -> str:
        """Auto-detect the domain/type of data"""
        columns_lower = [col.lower() for col in data_summary['columns']]
        
        finance_keywords = ['budget', 'cost', 'expense', 'revenue', 'amount', 'payment', 'vendor', 'invoice']
        hr_keywords = ['employee', 'staff', 'leave', 'performance', 'salary', 'department', 'hire', 'turnover']
        ops_keywords = ['process', 'efficiency', 'service', 'delivery', 'issue', 'ticket', 'procurement', 'operation']
        
        finance_score = sum(1 for keyword in finance_keywords if any(keyword in col for col in columns_lower))
        hr_score = sum(1 for keyword in hr_keywords if any(keyword in col for col in columns_lower))
        ops_score = sum(1 for keyword in ops_keywords if any(keyword in col for col in columns_lower))
        
        if finance_score >= hr_score and finance_score >= ops_score:
            return "Finance"
        elif hr_score >= ops_score:
            return "HR"
        elif ops_score > 0:
            return "Operations"
        else:
            return "General Government Data"
    
    def _parse_questions(self, questions_text: str) -> List[str]:
        """Parse the AI response to extract clean questions"""
        lines = questions_text.strip().split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            if line and any(line.startswith(str(i)) for i in range(1, 10)):
                question = line.split('.', 1)[-1].strip()
                if question and len(question) > 10:
                    questions.append(question)
        
        return questions
    
    def _get_fallback_questions(self) -> List[str]:
        """Fallback questions if AI generation fails"""
        return [
            "Main data patterns?",
            "Any outliers detected?",
            "Key trends identified?",
            "Budget variances?",
            "Performance metrics?"
        ]
class DataAnalysisEngine:
    def __init__(self):
        pass
    
    def execute_query_analysis(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Thực hiện phân tích dữ liệu thực tế dựa trên query"""
        results = {
            'direct_answers': [],
            'relevant_data': {},
            'statistical_insights': {},
            'data_found': False
        }
        

        query_lower = query.lower()

        relevant_columns = self._find_relevant_columns(df, query_lower)

        if 'how many' in query_lower or 'count' in query_lower:
            results.update(self._handle_count_queries(df, query_lower, relevant_columns))
        
        if 'average' in query_lower or 'mean' in query_lower:
            results.update(self._handle_average_queries(df, query_lower, relevant_columns))
        
        if 'sum' in query_lower or 'total' in query_lower:
            results.update(self._handle_sum_queries(df, query_lower, relevant_columns))

        results.update(self._search_specific_values(df, query_lower))
        
        return results
    
    def _find_relevant_columns(self, df: pd.DataFrame, query: str) -> List[str]:
        """Tìm columns liên quan đến query"""
        relevant_cols = []
        query_words = query.split()
        
        for col in df.columns:
            col_lower = col.lower()

            for word in query_words:
                if word in col_lower or col_lower in word:
                    relevant_cols.append(col)
                    break
        
        return relevant_cols
    
    def _handle_count_queries(self, df: pd.DataFrame, query: str, relevant_columns: List[str]) -> Dict:
        """Xử lý câu hỏi đếm số lượng"""
        results = {'count_results': []}
        

        for col in df.columns:
            if df[col].dtype == 'object':  # Text columns
                unique_values = df[col].unique()
                for value in unique_values:
                    if str(value).lower() in query:
                        count = len(df[df[col] == value])
                        results['count_results'].append({
                            'question': f"How many {col} = {value}",
                            'answer': count,
                            'column': col,
                            'value': value
                        })
                        results['data_found'] = True
        
        return results
    
    def _search_specific_values(self, df: pd.DataFrame, query: str) -> Dict:
        """Tìm kiếm giá trị cụ thể trong data"""
        results = {'value_matches': []}
        
        query_words = [word.strip('?.,!') for word in query.split()]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                for value in df[col].unique():
                    value_str = str(value).lower()
                    for word in query_words:
                        if word.lower() in value_str or value_str in word.lower():
                            count = len(df[df[col] == value])
                            results['value_matches'].append({
                                'column': col,
                                'value': value,
                                'count': count,
                                'percentage': round(count/len(df)*100, 1)
                            })
                            results['data_found'] = True
        
        return results

class TrustScoreCalculator:
    def __init__(self):
        self.base_score = 0.7
    
    def calculate_trust_score(self, query: str, data_coverage: float, 
                            response_specificity: float, source_quality: float) -> Dict[str, Any]:
        """Calculate comprehensive trust score"""
        
        weights = {
            'data_coverage': 0.4,
            'response_specificity': 0.3,
            'source_quality': 0.3
        }
        
        trust_score = (
            data_coverage * weights['data_coverage'] +
            response_specificity * weights['response_specificity'] +
            source_quality * weights['source_quality']
        )
        
        if trust_score >= 0.9:
            confidence_level = "Very High"
            color = "green"
        elif trust_score >= 0.8:
            confidence_level = "High"
            color = "lightgreen"
        elif trust_score >= 0.7:
            confidence_level = "Moderate"
            color = "orange"
        else:
            confidence_level = "Low"
            color = "red"
        
        return {
            'score': round(trust_score, 3),
            'confidence_level': confidence_level,
            'color': color,
            'components': {
                'data_coverage': round(data_coverage, 3),
                'response_specificity': round(response_specificity, 3),
                'source_quality': round(source_quality, 3)
            }
        }

class AuditTrail:
    def __init__(self):
        self.steps = []
    
    def add_step(self, step_type: str, description: str, data_used: List[str], 
                timestamp: str = None, details: Dict[str, Any] = None):
        """Add detailed step to audit trail"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        step = {
            'timestamp': timestamp,
            'step_type': step_type,
            'description': description,
            'data_used': data_used,
            'details': details or {},
            'step_id': hashlib.md5(f"{timestamp}{description}".encode()).hexdigest()[:8]
        }
        self.steps.append(step)
        return step['step_id']
    
    def add_data_analysis_step(self, df: pd.DataFrame, query: str, analysis_results: Dict):
        """Add detailed data analysis step"""

        if analysis_results is None:
            analysis_results = {}
        
        details = {
            'query_processed': query,
            'datasets_analyzed': list(df['source_dataset'].unique()) if 'source_dataset' in df.columns else ['Combined Dataset'],
            'total_records_processed': len(df),
            'columns_analyzed': list(df.columns),
            'data_quality_metrics': {
                'completeness': round(1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))), 3),
                'missing_values_by_column': df.isnull().sum().to_dict(),
                'data_types': {col: str(df[col].dtype) for col in df.columns}
            },

            'statistical_summary': analysis_results.get('summary_stats', {}).to_dict() if hasattr(analysis_results.get('summary_stats', {}), 'to_dict') else analysis_results.get('summary_stats', {}),
            'key_findings': self._extract_key_findings(df, query)
        }
        
        return self.add_step(
            "DETAILED_DATA_ANALYSIS", 
            f"Comprehensive analysis of {len(df)} records across {len(df.columns)} columns for query: '{query[:50]}...'",
            [f"Dataset with {len(df)} rows"],
            details=details
        )

    
    def add_ai_reasoning_step(self, query: str, context: str, ai_response: str, reasoning_process: Dict):
        """Add detailed AI reasoning step"""

        if reasoning_process is None:
            reasoning_process = {}
        
        details = {
            'original_query': query,
            'context_provided': context[:500] + "..." if len(context) > 500 else context,
            'reasoning_steps': reasoning_process.get('steps', []),
            'data_points_referenced': reasoning_process.get('data_points', []),
            'confidence_factors': reasoning_process.get('confidence_factors', {}),
            'limitations_identified': reasoning_process.get('limitations', []),
            'assumptions_made': reasoning_process.get('assumptions', []),
            'response_generated': ai_response[:200] + "..." if len(ai_response) > 200 else ai_response
        }
        
        return self.add_step(
            "AI_REASONING_PROCESS",
            f"AI reasoning for query analysis with {len(reasoning_process.get('steps', []))} logical steps",
            ["User query", "Dataset context", "Statistical analysis"],
            details=details
        )

    def add_trust_calculation_step(self, trust_components: Dict, calculation_method: Dict):
        """Add detailed trust score calculation step"""
        details = {
            'trust_components': trust_components,
            'calculation_method': calculation_method,
            'component_weights': {
                'data_coverage': 0.4,
                'response_specificity': 0.3,
                'source_quality': 0.3
            },
            'final_score_calculation': self._show_trust_calculation(trust_components)
        }
        
        return self.add_step(
            "TRUST_SCORE_CALCULATION",
            f"Trust score calculated: {trust_components.get('score', 0)} based on multiple factors",
            ["Data quality metrics", "Response analysis", "Source reliability"],
            details=details
        )
    
    def _extract_key_findings(self, df: pd.DataFrame, query: str) -> List[str]:
        """Extract key findings from data analysis"""
        findings = []
        
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            findings.append(f"Missing data detected in {missing_data[missing_data > 0].count()} columns")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            findings.append(f"Numerical analysis performed on {len(numeric_cols)} columns")
        
        query_words = query.lower().split()
        matching_columns = [col for col in df.columns if any(word in col.lower() for word in query_words)]
        if matching_columns:
            findings.append(f"Query-relevant columns identified: {', '.join(matching_columns[:3])}")
        
        return findings
    
    def _show_trust_calculation(self, components: Dict) -> Dict:
        """Show detailed trust score calculation"""
        return {
            'formula': '(data_coverage * 0.4) + (response_specificity * 0.3) + (source_quality * 0.3)',
            'calculation': f"({components.get('data_coverage', 0)} * 0.4) + ({components.get('response_specificity', 0)} * 0.3) + ({components.get('source_quality', 0)} * 0.3)",
            'result': components.get('score', 0)
        }
    
    def get_trail(self) -> List[Dict]:
        """Get complete audit trail"""
        return self.steps

class DataExporter:
    @staticmethod
    def export_conversation_history(conversation_history: List[Dict]) -> str:
        """Export conversation history to CSV format"""
        if not conversation_history:
            return ""
        
        df = pd.DataFrame(conversation_history)
        return df.to_csv(index=False)
    
    @staticmethod
    def export_conversation_history_json(conversation_history: List[Dict]) -> str:
        """Export conversation history to JSON format"""
        return json.dumps(conversation_history, indent=2, ensure_ascii=False)
    
    @staticmethod
    def export_audit_trail(audit_trail: List[Dict]) -> str:
        """Export audit trail to CSV format"""
        if not audit_trail:
            return ""
        
        flattened_data = []
        for step in audit_trail:
            flattened_data.append({
                'step_id': step['step_id'],
                'timestamp': step['timestamp'],
                'step_type': step['step_type'],
                'description': step['description'],
                'data_used': '; '.join(step['data_used'])
            })
        
        df = pd.DataFrame(flattened_data)
        return df.to_csv(index=False)
    
    @staticmethod
    def export_audit_trail_json(audit_trail: List[Dict]) -> str:
        """Export audit trail to JSON format"""
        return json.dumps(audit_trail, indent=2, ensure_ascii=False)

class ConversationalAI:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.trust_calculator = TrustScoreCalculator()
        self.audit = AuditTrail()
    
    def analyze_data(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Phân tích dữ liệu chi tiết dựa trên query"""
        

        analysis_engine = DataAnalysisEngine()
        query_results = analysis_engine.execute_query_analysis(df, query)
        
        analysis = {
            'summary_stats': df.describe() if not df.empty else None,
            'column_info': {col: str(df[col].dtype) for col in df.columns} if not df.empty else {},
            'missing_values': df.isnull().sum().to_dict() if not df.empty else {},
            'unique_values': {col: df[col].nunique() for col in df.columns if df[col].dtype == 'object'} if not df.empty else {},
            'query_analysis': query_results  # Thêm kết quả phân tích query
        }
    def generate_response(self, query: str, data_analysis: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate AI response with detailed audit trail"""
        
        try:
 
            if data_analysis is None:
                data_analysis = {}
            
            step_id_analysis = self.audit.add_data_analysis_step(df, query, data_analysis)
            
            context = self._create_context(data_analysis, df)
            
            reasoning_process = self._create_reasoning_process(query, df, data_analysis)
            
   
            if reasoning_process is None:
                reasoning_process = {
                    'steps': [],
                    'data_points': [],
                    'confidence_factors': {},
                    'limitations': [],
                    'assumptions': []
                }
            
            prompt = self._create_detailed_prompt(query, context, reasoning_process)
            

            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a government data analyst AI. Provide accurate, factual responses based only on the provided data. Always cite specific data points and be transparent about limitations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=250,
                    temperature=0.1
                )
                

                if response and response.choices and len(response.choices) > 0:
                    ai_response = response.choices[0].message.content
                else:
                    ai_response = "Unable to generate response from AI model."
                    
            except Exception as api_error:
                ai_response = f"AI API Error: {str(api_error)}"
            
            step_id_reasoning = self.audit.add_ai_reasoning_step(query, context, ai_response, reasoning_process)
            
            data_coverage = self._calculate_data_coverage(query, df)
            response_specificity = self._calculate_response_specificity(ai_response)
            source_quality = self._calculate_source_quality(df)
            
            trust_components = {
                'data_coverage': data_coverage,
                'response_specificity': response_specificity,
                'source_quality': source_quality
            }
            
            calculation_method = {
                'data_coverage_method': 'Keyword overlap between query and dataset columns',
                'response_specificity_method': 'Response length and numerical content analysis',
                'source_quality_method': 'Data completeness and consistency metrics'
            }
            
            trust_score = self.trust_calculator.calculate_trust_score(
                query, data_coverage, response_specificity, source_quality
            )
            

            if trust_score is None:
                trust_score = {
                    'score': 0.0,
                    'confidence_level': 'Low',
                    'color': 'red',
                    'components': trust_components
                }
            else:
                trust_score['components'] = trust_components
            
            step_id_trust = self.audit.add_trust_calculation_step(trust_components, calculation_method)
            
            return {
                'response': ai_response,
                'trust_score': trust_score,
                'audit_trail': self.audit.get_trail(),
                'data_sources_used': list(df['source_dataset'].unique()) if 'source_dataset' in df.columns else ['Combined Dataset'],
                'reasoning_process': reasoning_process
            }
            
        except Exception as e:
            # Improved error handling
            error_trust_score = {
                'score': 0.0, 
                'confidence_level': 'Error', 
                'color': 'red',
                'components': {
                    'data_coverage': 0.0,
                    'response_specificity': 0.0,
                    'source_quality': 0.0
                }
            }
            return {
                'response': f"Error generating response: {str(e)}",
                'trust_score': error_trust_score,
                'audit_trail': self.audit.get_trail(),
                'data_sources_used': [],
                'reasoning_process': {
                    'error': str(e),
                    'steps': [],
                    'data_points': [],
                    'confidence_factors': {},
                    'limitations': [f"Error occurred: {str(e)}"],
                    'assumptions': []
                }
            }

    def _create_reasoning_process(self, query: str, df: pd.DataFrame, analysis: Dict) -> Dict[str, Any]:
        
        """Create detailed reasoning process for AI analysis"""
        
        reasoning = {
            'steps': [],
            'data_points': [],
            'confidence_factors': {},
            'limitations': [],
            'assumptions': []
        }
        
        reasoning['steps'].append({
            'step_number': 1,
            'action': 'Query Analysis',
            'description': f"Parsed user query: '{query}' to identify key information needs",
            'outcome': f"Identified {len(query.split())} keywords for data matching"
        })
        
        query_words = set(query.lower().split())
        column_words = set(' '.join(df.columns).lower().split())
        matches = query_words.intersection(column_words)
        
        reasoning['steps'].append({
            'step_number': 2,
            'action': 'Data Matching',
            'description': f"Matched query keywords with available data columns",
            'outcome': f"Found {len(matches)} direct matches: {list(matches) if matches else 'No direct matches'}"
        })
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        reasoning['steps'].append({
            'step_number': 3,
            'action': 'Statistical Analysis',
            'description': f"Performed statistical analysis on {len(numeric_cols)} numerical columns",
            'outcome': f"Generated descriptive statistics and identified data patterns"
        })
        
        reasoning['data_points'] = [
            f"Total records analyzed: {len(df)}",
            f"Columns available: {len(df.columns)}",
            f"Data completeness: {round(1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))), 2) * 100}%"
        ]
        
        reasoning['confidence_factors'] = {
            'data_availability': 'High' if len(df) > 100 else 'Medium' if len(df) > 10 else 'Low',
            'query_data_match': 'High' if matches else 'Medium',
            'data_quality': 'High' if df.isnull().sum().sum() < len(df) * 0.1 else 'Medium'
        }
        
        if df.empty:
            reasoning['limitations'].append("No data available for analysis")
        if not matches:
            reasoning['limitations'].append("No direct keyword matches between query and data columns")
        if df.isnull().sum().sum() > len(df) * 0.2:
            reasoning['limitations'].append("Significant missing data may affect accuracy")
        
        reasoning['assumptions'].append("Data provided is accurate and up-to-date")
        reasoning['assumptions'].append("User query relates to available dataset columns")
        if len(df) < 1000:
            reasoning['assumptions'].append("Small dataset size - results may not be fully representative")
        
        return reasoning

    def _create_detailed_prompt(self, query: str, context: str, reasoning: Dict) -> str:
        """Tạo prompt với nhấn mạnh sử dụng kết quả phân tích thực tế"""
        return f"""
        You are analyzing government data. Use the DIRECT DATA ANALYSIS RESULTS provided below to answer the user's question.

        IMPORTANT: If direct analysis results are provided, use them as your primary source. Be specific with numbers and findings.

        Data Analysis Results:
        {context}

        User Question: {query}

        Instructions:
        - If you see "DIRECT DATA ANALYSIS RESULTS" above, use those exact numbers and findings
        - Answer with specific data points from the analysis
        - If count results are shown, state the exact counts
        - If value matches are found, mention them specifically
        - Keep response under 100 words but be precise with data
        - If no direct results, acknowledge what data is available and suggest related insights
        """

    def _create_context(self, analysis: Dict, df: pd.DataFrame) -> str:
        """Tạo context chi tiết với kết quả phân tích thực tế"""
        context_parts = []
        
        if not df.empty:
            context_parts.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")
            context_parts.append(f"Columns: {', '.join(df.columns)}")
            

            query_analysis = analysis.get('query_analysis', {})
            
            if query_analysis.get('data_found'):
                context_parts.append("\n=== DIRECT DATA ANALYSIS RESULTS ===")
                
 
                if query_analysis.get('count_results'):
                    context_parts.append("COUNT RESULTS:")
                    for result in query_analysis['count_results']:
                        context_parts.append(f"- {result['question']}: {result['answer']}")
                

                if query_analysis.get('value_matches'):
                    context_parts.append("VALUE MATCHES:")
                    for match in query_analysis['value_matches']:
                        context_parts.append(f"- {match['column']} '{match['value']}': {match['count']} records ({match['percentage']}%)")
            

            if not query_analysis.get('data_found'):
                sample_data = df.head(5).to_string()
                context_parts.append(f"\nSample data:\n{sample_data}")
        
        return "\n".join(context_parts)
    
    def _calculate_data_coverage(self, query: str, df: pd.DataFrame) -> float:
        """Tính toán coverage dựa trên khả năng trả lời query"""
        if df.empty:
            return 0.0
        

        analysis_engine = DataAnalysisEngine()
        query_results = analysis_engine.execute_query_analysis(df, query)
        

        if query_results.get('data_found'):
            return 0.95  
        

        query_words = set(query.lower().split())
        column_words = set(' '.join(df.columns).lower().split())
        

        content_words = set()
        for col in df.columns:
            if df[col].dtype == 'object':
                content_words.update(' '.join(df[col].astype(str).unique()).lower().split())
        
        all_data_words = column_words.union(content_words)
        overlap = len(query_words.intersection(all_data_words))
        coverage = min(overlap / len(query_words) if query_words else 0, 1.0)
        
        return max(coverage, 0.3)  # Minimum coverage

    
    def _calculate_response_specificity(self, response: str) -> float:
        """Calculate response specificity"""
        word_count = len(response.split())
        has_numbers = any(char.isdigit() for char in response)
        
        specificity = min(word_count / 100, 1.0)
        if has_numbers:
            specificity = min(specificity + 0.2, 1.0)
        
        return max(specificity, 0.6)
    
    def _calculate_source_quality(self, df: pd.DataFrame) -> float:
        """Calculate source data quality"""
        if df.empty:
            return 0.0
        
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        quality = completeness
        
        return max(quality, 0.7)

def main():
    if not st.session_state.logged_in:
        show_login()
        return
    
    with st.sidebar:
        st.image("australian-government.png", width=200)
        st.markdown("---")
        
        st.write(f"👤 **{st.session_state.username}**")
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
        st.markdown("---")
    
    main_app()

def show_login():
    st.title("🔐 Government Data AI")
    
    db = UserDB()
    
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            with st.form("login_form"):
                st.markdown("### Demo Account")
                st.info("Username: **user** | Password: **user**")
                
                username = st.text_input("Username", value="user")
                password = st.text_input("Password", type="password", value="user")
                
                if st.form_submit_button("🚀 Login", type="primary"):
                    if db.verify_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials!")
    
    with tab2:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            with st.form("register_form"):
                st.markdown("### Create New Account")
                
                new_username = st.text_input("Username", placeholder="Username (3-20 characters)")
                new_full_name = st.text_input("Full Name", placeholder="Enter your full name")
                new_password = st.text_input("Password", type="password", placeholder="Password (minimum 6 characters)")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                
                if st.form_submit_button("📝 Register", type="primary"):
                    if not all([new_username, new_password, confirm_password]):
                        st.error("❌ Please fill in all information!")
                    elif len(new_username) < 3 or len(new_username) > 20:
                        st.error("❌ Username must be 3-20 characters!")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters!")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords do not match!")
                    else:
                        if db.create_user(new_username, new_password, new_full_name):
                            st.success("✅ Registration successful! Please switch to Login tab.")
                            st.balloons()
                        else:
                            st.error("❌ Username already exists!")

def main_app():
    
    st.title("🏛️ Government Data Conversational AI")
    st.markdown("### Accurate and Trustworthy Chatbot for Data Interactions")
    
    st.sidebar.header("Configuration")
    st.sidebar.success("🔑 OpenAI API: Connected")
    
    st.header("📊 Data Upload")
    uploaded_files = st.file_uploader(
        "Upload your government datasets (CSV, Excel)",
        accept_multiple_files=True,
        type=['csv', 'xlsx', 'xls'],
        help="Upload multiple files from different departments. Files will be combined intelligently."
    )
    
    if uploaded_files:
        processor = DataProcessor()
        profile_generator = ProfileReportGenerator()
        datasets = {}
        
        current_files_hash = processor.get_files_hash(uploaded_files)
        files_changed = current_files_hash != st.session_state.uploaded_files_hash
        
        if files_changed:
            st.session_state.suggested_questions = []
            st.session_state.dataset_questions = {}
            st.session_state.profile_reports = {}
            st.session_state.uploaded_files_hash = current_files_hash
        
        for uploaded_file in uploaded_files:
            df, metadata = processor.load_file(uploaded_file)
            if df is not None:
                datasets[uploaded_file.name] = df
                st.session_state.data_sources[uploaded_file.name] = metadata
        
        if datasets:
            combined_df = processor.combine_datasets(datasets)
            
            st.metric("📁 Data Sources", len(datasets))
            
            st.header("📋 Data Sources & Analysis")
            
            if len(datasets) == 1:
                dataset_name = list(datasets.keys())[0]
                df = datasets[dataset_name]
                metadata = st.session_state.data_sources[dataset_name]
                
                with st.expander(f"📄 {dataset_name}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Records", metadata['rows'])
                    with col2:
                        st.metric("Columns", len(metadata['columns']))
                    with col3:
                        file_size = f"{metadata['size'] / 1024:.1f} KB" if metadata['size'] < 1024*1024 else f"{metadata['size'] / (1024*1024):.1f} MB"
                        st.metric("Size", file_size)
                    
                    tab1, tab2, tab3 = st.tabs(["📋 Preview", "📊 Profile Report", "ℹ️ Info"])
                    
                    with tab1:
                        st.markdown("**Column Names:**")
                        st.write(", ".join(metadata['columns']))
                        
                        st.markdown("**Data Preview:**")
                        st.dataframe(df.head(10))
                    
                    with tab2:
                        if st.button(f"🔄 Generate Profile Report", key=f"profile_{dataset_name}"):
                            with st.spinner("Generating comprehensive data profile..."):
                                profile = profile_generator.generate_profile_report(df, dataset_name, minimal=True)
                                if profile:
                                    st.session_state.profile_reports[dataset_name] = profile
                        
                        if dataset_name in st.session_state.profile_reports:
                            profile_generator.display_profile_report(
                                st.session_state.profile_reports[dataset_name], 
                                key=f"profile_display_{dataset_name}"
                            )
                    
                    with tab3:
                        st.markdown("**Dataset Information:**")
                        st.write(f"- **Filename:** {metadata['filename']}")
                        st.write(f"- **Upload Time:** {metadata['upload_time']}")
                        st.write(f"- **Records:** {metadata['rows']:,}")
                        st.write(f"- **Columns:** {len(metadata['columns'])}")
                        st.write(f"- **File Size:** {file_size}")
            
            else:
                for dataset_name, df in datasets.items():
                    metadata = st.session_state.data_sources[dataset_name]
                    
                    with st.expander(f"📄 {dataset_name}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Records", metadata['rows'])
                        with col2:
                            st.metric("Columns", len(metadata['columns']))
                        with col3:
                            file_size = f"{metadata['size'] / 1024:.1f} KB" if metadata['size'] < 1024*1024 else f"{metadata['size'] / (1024*1024):.1f} MB"
                            st.metric("Size", file_size)
                        
                        tab1, tab2, tab3 = st.tabs(["📋 Preview", "📊 Profile Report", "ℹ️ Info"])
                        
                        with tab1:
                            st.markdown("**Column Names:**")
                            st.write(", ".join(metadata['columns']))
                            
                            st.markdown("**Data Preview:**")
                            st.dataframe(df.head(10))
                        
                        with tab2:
                            if st.button(f"🔄 Generate Profile Report", key=f"profile_{dataset_name}"):
                                with st.spinner(f"Generating profile for {dataset_name}..."):
                                    profile = profile_generator.generate_profile_report(df, dataset_name, minimal=True)
                                    if profile:
                                        st.session_state.profile_reports[dataset_name] = profile
                                        st.success("✅ Profile report generated successfully!")
                            
                            if dataset_name in st.session_state.profile_reports:
                                profile_generator.display_profile_report(
                                    st.session_state.profile_reports[dataset_name], 
                                    key=f"profile_display_{dataset_name}"
                                )
                        
                        with tab3:
                            st.markdown("**Dataset Information:**")
                            st.write(f"- **Filename:** {metadata['filename']}")
                            st.write(f"- **Upload Time:** {metadata['upload_time']}")
                            st.write(f"- **Records:** {metadata['rows']:,}")
                            st.write(f"- **Columns:** {len(metadata['columns'])}")
                            st.write(f"- **File Size:** {file_size}")
            
            if not st.session_state.suggested_questions:
                with st.spinner("🤖 AI is analyzing your datasets and their connections to generate relevant questions..."):
                    try:
                        question_generator = QuestionGenerator(OPENAI_API_KEY)
                        
                        all_questions = question_generator.generate_questions_for_datasets(datasets)
                        st.session_state.dataset_questions = all_questions
                        
                        selected_questions = question_generator.select_top_questions(all_questions, 5)
                        st.session_state.suggested_questions = selected_questions
                        
                    except Exception as e:
                        st.warning(f"Unable to generate AI questions: {str(e)}")
                        st.session_state.suggested_questions = [
                            "What are the main patterns across all datasets?",
                            "Are there any correlations between different data sources?",
                            "What outliers need attention?",
                            "How do the datasets complement each other?",
                            "What insights emerge from combining these datasets?"
                        ]
            
            st.header("❓ Question Guidance")
            
            if st.session_state.suggested_questions:
                st.markdown("**🤖 AI-Generated Questions Based on Your Data:**")
                
                if len(datasets) > 1:
                    st.info(f"📊 These questions were generated by analyzing {len(datasets)} datasets and their connections. Cross-dataset questions are prioritized to help you find relationships between different data sources.")
                else:
                    st.info("📊 These questions were automatically generated by analyzing your uploaded dataset.")
                
                num_questions = len(st.session_state.suggested_questions)
                
                if num_questions <= 3:
                    cols = st.columns(num_questions)
                    for i, question in enumerate(st.session_state.suggested_questions):
                        with cols[i]:
                            if st.button(f"📝 {question}", key=f"ai_question_{i}"):
                                st.session_state.user_question = question
                else:
                    questions_per_row = 3
                    for row in range(0, num_questions, questions_per_row):
                        row_questions = st.session_state.suggested_questions[row:row + questions_per_row]
                        cols = st.columns(len(row_questions))
                        
                        for i, question in enumerate(row_questions):
                            with cols[i]:
                                if st.button(f"📝 {question}", key=f"ai_question_{row + i}"):
                                    st.session_state.user_question = question
                        
            if st.session_state.suggested_questions:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Generate New Questions", help="Generate a new set of 5 questions"):
                        if st.session_state.dataset_questions:
                            question_generator = QuestionGenerator(OPENAI_API_KEY)
                            selected_questions = question_generator.select_top_questions(st.session_state.dataset_questions, 5)
                            st.session_state.suggested_questions = selected_questions
                            st.rerun()
                
                with col2:
                    if st.button("🎲 Shuffle Questions", help="Randomly select 5 questions from all available"):
                        if st.session_state.dataset_questions:
                            all_questions_flat = []
                            for questions in st.session_state.dataset_questions.values():
                                all_questions_flat.extend(questions)
                            if len(all_questions_flat) >= 5:
                                st.session_state.suggested_questions = random.sample(all_questions_flat, 5)
                            else:
                                st.session_state.suggested_questions = all_questions_flat
                            st.rerun()
            
            st.header("💬 Ask Your Question")
            user_question = st.text_area(
                "Enter your question about the data:",
                value=st.session_state.get('user_question', ''),
                height=100,
                help="Ask specific questions about your data. Be clear about what you want to know."
            )
            
            if st.button("🔍 Analyze Data", type="primary") and user_question:
                with st.spinner("Analyzing data and generating response..."):
                    try:
                        ai = ConversationalAI(OPENAI_API_KEY)
                        
                        analysis = ai.analyze_data(combined_df, user_question)
                        
                        result = ai.generate_response(user_question, analysis, combined_df)
                        
                        st.header("🤖 AI Response")
                        
                        trust_info = result['trust_score']
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Response:** {result['response']}")
                        
                        with col2:
                            st.markdown(f"### Trust Score")
                            st.markdown(f"<div style='background-color: {trust_info['color']}; padding: 10px; border-radius: 5px; text-align: center;'>"
                                      f"<strong>{trust_info['score']}</strong><br>"
                                      f"<small>{trust_info['confidence_level']}</small></div>", 
                                      unsafe_allow_html=True)
                        
                        with st.expander("🔍 Trust Score Breakdown"):
                            st.write("**Components:**")
                            if 'components' in trust_info:
                                for component, value in trust_info['components'].items():
                                    st.write(f"- {component.replace('_', ' ').title()}: {value}")
                            else:
                                st.write("Trust score components not available")
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.header("📋 Audit Trail")
                        with col2:
                            exporter = DataExporter()
                            audit_csv = exporter.export_audit_trail(result['audit_trail'])
                            if audit_csv:
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=audit_csv,
                                    file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    help="Download audit trail as CSV file"
                                )
                        with col3:
                            audit_json = exporter.export_audit_trail_json(result['audit_trail'])
                            if audit_json:
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=audit_json,
                                    file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    help="Download audit trail as JSON file"
                                )

                        audit_trail = result['audit_trail']
                        for i, step in enumerate(audit_trail):
                            with st.expander(f"Step {i+1}: {step['step_type']} - {step['description'][:50]}...", expanded=False):
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Timestamp:** {step['timestamp']}")
                                    st.write(f"**Step ID:** {step['step_id']}")
                                with col2:
                                    st.write(f"**Type:** {step['step_type']}")
                                    st.write(f"**Data Used:** {', '.join(step['data_used'])}")
                                
                                st.write(f"**Description:** {step['description']}")
                                
                                if 'details' in step and step['details']:
                                    st.markdown("### 🔍 Detailed Analysis")
                                    
                                    details = step['details']
                                    
                                    if step['step_type'] == 'DETAILED_DATA_ANALYSIS':
                                        st.markdown("**📊 Data Quality Metrics:**")
                                        quality = details.get('data_quality_metrics', {})
                                        st.write(f"- Completeness: {quality.get('completeness', 'N/A')}")
                                        st.write(f"- Total Records: {details.get('total_records_processed', 'N/A')}")
                                        st.write(f"- Columns Analyzed: {len(details.get('columns_analyzed', []))}")
                                        
                                        if quality.get('missing_values_by_column'):
                                            st.markdown("**Missing Values by Column:**")
                                            missing_df = pd.DataFrame(list(quality['missing_values_by_column'].items()), 
                                                                    columns=['Column', 'Missing Count'])
                                            st.dataframe(missing_df[missing_df['Missing Count'] > 0])
                                        
                                        if details.get('key_findings'):
                                            st.markdown("**Key Findings:**")
                                            for finding in details['key_findings']:
                                                st.write(f"- {finding}")
                                    
                                    elif step['step_type'] == 'AI_REASONING_PROCESS':
                                        st.markdown("**🧠 Reasoning Steps:**")
                                        for reasoning_step in details.get('reasoning_steps', []):
                                            st.write(f"**Step {reasoning_step.get('step_number')}:** {reasoning_step.get('action')}")
                                            st.write(f"   - {reasoning_step.get('description')}")
                                            st.write(f"   - Outcome: {reasoning_step.get('outcome')}")
                                        
                                        st.markdown("**📈 Confidence Factors:**")
                                        for factor, level in details.get('confidence_factors', {}).items():
                                            st.write(f"- {factor.replace('_', ' ').title()}: {level}")
                                        
                                        if details.get('limitations_identified'):
                                            st.markdown("**⚠️ Limitations Identified:**")
                                            for limitation in details['limitations_identified']:
                                                st.write(f"- {limitation}")
                                        
                                        if details.get('assumptions_made'):
                                            st.markdown("**💭 Assumptions Made:**")
                                            for assumption in details['assumptions_made']:
                                                st.write(f"- {assumption}")
                                    
                                    elif step['step_type'] == 'TRUST_SCORE_CALCULATION':
                                        st.markdown("**🎯 Trust Score Breakdown:**")
                                        calc = details.get('final_score_calculation', {})
                                        st.write(f"**Formula:** {calc.get('formula', 'N/A')}")
                                        st.write(f"**Calculation:** {calc.get('calculation', 'N/A')}")
                                        st.write(f"**Result:** {calc.get('result', 'N/A')}")
                                        
                                        st.markdown("**Component Weights:**")
                                        weights = details.get('component_weights', {})
                                        for component, weight in weights.items():
                                            st.write(f"- {component.replace('_', ' ').title()}: {weight}")

                        st.header("📊 Data Sources Used in Analysis")
                        sources_used = result['data_sources_used']
                        for source in sources_used:
                            if source in st.session_state.data_sources:
                                metadata = st.session_state.data_sources[source]
                                with st.expander(f"📄 {source}"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"**Records Used:** {metadata['rows']}")
                                        st.write(f"**Columns Available:** {len(metadata['columns'])}")
                                    with col2:
                                        st.write(f"**Upload Time:** {metadata['upload_time']}")
                                        file_size = f"{metadata['size'] / 1024:.1f} KB" if metadata['size'] < 1024*1024 else f"{metadata['size'] / (1024*1024):.1f} MB"
                                        st.write(f"**File Size:** {file_size}")
                        
                        st.session_state.conversation_history.append({
                            'question': user_question,
                            'response': result['response'],
                            'trust_score': trust_info['score'],
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.write("Please check your API key configuration.")
            
            if st.session_state.conversation_history:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.header("📜 Conversation History")
                with col2:
                    exporter = DataExporter()
                    csv_data = exporter.export_conversation_history(st.session_state.conversation_history)
                    if csv_data:
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download conversation history as CSV file"
                        )
                with col3:
                    json_data = exporter.export_conversation_history_json(st.session_state.conversation_history)
                    if json_data:
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_data,
                            file_name=f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            help="Download conversation history as JSON file"
                        )
                
                for i, conv in enumerate(reversed(st.session_state.conversation_history)):
                    with st.expander(f"Q{len(st.session_state.conversation_history)-i}: {conv['question'][:100]}..."):
                        st.write(f"**Question:** {conv['question']}")
                        st.write(f"**Response:** {conv['response']}")
                        st.write(f"**Trust Score:** {conv['trust_score']}")
                        st.write(f"**Time:** {conv['timestamp']}")
    else:
        st.info("👆 Please upload your government datasets to begin analysis.")
        
        st.header("📋 Expected Data Formats")
        
        example_data = {
            "Finance Data": pd.DataFrame({
                'Department': ['Finance', 'HR', 'Operations'],
                'Budget_Allocated': [100000, 80000, 120000],
                'Budget_Spent': [95000, 75000, 115000],
                'Variance': [5000, 5000, 5000]
            }),
            "HR Data": pd.DataFrame({
                'Employee_ID': ['EMP001', 'EMP002', 'EMP003'],
                'Department': ['Finance', 'HR', 'Operations'],
                'Leave_Days': [15, 20, 10],
                'Performance_Score': [8.5, 9.0, 7.8]
            }),
            "Operations Data": pd.DataFrame({
                'Process_ID': ['PROC001', 'PROC002', 'PROC003'],
                'Department': ['Operations', 'Finance', 'HR'],
                'Efficiency_Score': [85, 92, 88],
                'Issues_Count': [2, 1, 3]
            })
        }
        
        for name, df in example_data.items():
            with st.expander(f"📊 Example: {name}"):
                st.dataframe(df)

if __name__ == "__main__":
    main()
