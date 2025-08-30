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

# YData Profiling imports
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

# Page configuration
st.set_page_config(
    page_title="Government Data Conversational AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load API key from secrets
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

class DataProcessor:
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls']
    
    def load_file(self, uploaded_file):
        """Load and process uploaded file"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                return None, f"Unsupported file format: {file_extension}"
            
            # Add metadata
            metadata = {
                'filename': uploaded_file.name,
                'size': uploaded_file.size,
                'columns': list(df.columns),
                'rows': len(df),
                'upload_time': datetime.now().isoformat(),
                'dataframe': df  # Store the actual dataframe
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
            # Performance optimization
            if len(df) > 1000:
                df_sample = df.sample(n=1000, random_state=42)
                st.warning(f"⚡ Using sample of 1000 rows from {len(df)} total rows for faster processing.")
            else:
                df_sample = df
                
            # Simple and compatible ProfileReport
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
            # Try using streamlit-ydata-profiling first
            st_profile_report(profile, navbar=True, key=key)
            
        except ImportError:
            st.warning("📦 Installing streamlit-ydata-profiling recommended for better display")
            try:
                # Fallback to HTML
                profile_html = profile.to_html()
                st.components.v1.html(profile_html, height=800, scrolling=True)
            except Exception as e:
                st.error(f"❌ Cannot display profile report: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Display error: {str(e)}")

# [Previous classes remain the same - DatasetConnectionAnalyzer, QuestionGenerator, TrustScoreCalculator, AuditTrail, ConversationalAI]
# ... [keeping all existing classes unchanged] ...

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
        
        # Find common columns
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                df1, df2 = datasets[name1], datasets[name2]
                common_cols = set(df1.columns).intersection(set(df2.columns))
                common_cols.discard('source_dataset')  # Ignore our added column
                
                if common_cols:
                    connections['common_columns'][f"{name1}_{name2}"] = list(common_cols)
        
        # Find potential join keys (ID-like columns)
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                df1, df2 = datasets[name1], datasets[name2]
                
                # Look for ID-like columns
                id_cols_1 = [col for col in df1.columns if 'id' in col.lower() or 'code' in col.lower()]
                id_cols_2 = [col for col in df2.columns if 'id' in col.lower() or 'code' in col.lower()]
                
                if id_cols_1 and id_cols_2:
                    connections['potential_joins'][f"{name1}_{name2}"] = {
                        'dataset1_keys': id_cols_1,
                        'dataset2_keys': id_cols_2
                    }
        
        # Analyze domain relationships
        connections['domain_relationships'] = self._analyze_domain_relationships(datasets)
        
        return connections
    
    def _analyze_domain_relationships(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Analyze relationships between different data domains"""
        relationships = {}
        
        for name, df in datasets.items():
            columns_lower = [col.lower() for col in df.columns]
            
            # Categorize dataset
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
        
        # Generate questions for each individual dataset
        for name, df in datasets.items():
            try:
                questions = self._generate_single_dataset_questions(df, name)
                all_questions[name] = questions
            except Exception as e:
                st.warning(f"Could not generate questions for {name}: {str(e)}")
                all_questions[name] = self._get_fallback_questions()
        
        # Generate cross-dataset questions if multiple datasets
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
        # Collect all questions
        question_pool = []
        
        for dataset_name, questions in all_questions.items():
            for question in questions:
                question_pool.append({
                    'question': question,
                    'source': dataset_name,
                    'priority': self._calculate_question_priority(question, dataset_name)
                })
        
        # Sort by priority and mix
        question_pool = sorted(question_pool, key=lambda x: x['priority'], reverse=True)
        
        # Select diverse questions (not all from same source)
        selected = []
        used_sources = set()
        
        # First pass: select high priority questions from different sources
        for item in question_pool:
            if len(selected) >= num_questions:
                break
            if item['source'] not in used_sources or len(used_sources) == len(all_questions):
                selected.append(item['question'])
                used_sources.add(item['source'])
        
        # Second pass: fill remaining slots randomly
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

        Generate questions focusing on key insights, trends, and outliers specific to this dataset.
        Format: Return exactly 3 questions, each on a new line, numbered 1-3.
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
        
        # Create summary of datasets
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

        Format: Return exactly 4 questions, each on a new line, numbered 1-4.
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
        
        # Cross-dataset questions get higher priority
        if source == 'cross_dataset':
            priority += 0.5
        
        # Questions with certain keywords get higher priority
        high_priority_keywords = ['trend', 'outlier', 'compare', 'correlation', 'pattern', 'efficiency', 'budget']
        for keyword in high_priority_keywords:
            if keyword.lower() in question.lower():
                priority += 0.2
                break
        
        # Questions with numbers/metrics get slight boost
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
        
        # Identify potential key fields
        analysis['potential_id_fields'] = [col for col in df.columns if 'id' in col.lower() or 'code' in col.lower()]
        analysis['potential_amount_fields'] = [col for col in df.columns if any(keyword in col.lower() for keyword in ['amount', 'cost', 'budget', 'salary', 'revenue', 'expense', 'price', 'value'])]
        analysis['potential_date_fields'] = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day'])]
        
        return analysis
    
    def _detect_data_domain(self, data_summary: Dict) -> str:
        """Auto-detect the domain/type of data"""
        columns_lower = [col.lower() for col in data_summary['columns']]
        
        # Count domain-specific keywords
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
                # Remove numbering and clean up
                question = line.split('.', 1)[-1].strip()
                if question and len(question) > 10:  # Filter out very short responses
                    questions.append(question)
        
        return questions
    
    def _get_fallback_questions(self) -> List[str]:
        """Fallback questions if AI generation fails"""
        return [
            "What are the main patterns in this dataset?",
            "Are there any outliers or unusual values that need attention?",
            "What trends can be identified in the data?"
        ]

class TrustScoreCalculator:
    def __init__(self):
        self.base_score = 0.7
    
    def calculate_trust_score(self, query: str, data_coverage: float, 
                            response_specificity: float, source_quality: float) -> Dict[str, Any]:
        """Calculate comprehensive trust score"""
        
        # Weight factors
        weights = {
            'data_coverage': 0.4,
            'response_specificity': 0.3,
            'source_quality': 0.3
        }
        
        # Calculate weighted score
        trust_score = (
            data_coverage * weights['data_coverage'] +
            response_specificity * weights['response_specificity'] +
            source_quality * weights['source_quality']
        )
        
        # Determine confidence level
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
                timestamp: str = None):
        """Add step to audit trail"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        step = {
            'timestamp': timestamp,
            'step_type': step_type,
            'description': description,
            'data_used': data_used,
            'step_id': hashlib.md5(f"{timestamp}{description}".encode()).hexdigest()[:8]
        }
        self.steps.append(step)
        return step['step_id']
    
    def get_trail(self) -> List[Dict]:
        """Get complete audit trail"""
        return self.steps

class ConversationalAI:
    def __init__(self, api_key: str):
        # Updated for OpenAI v1.0+
        self.client = OpenAI(api_key=api_key)
        self.trust_calculator = TrustScoreCalculator()
        self.audit = AuditTrail()
    
    def analyze_data(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze data based on user query"""
        
        # Add audit step
        self.audit.add_step("DATA_ANALYSIS", f"Analyzing data for query: {query}", 
                           [f"Dataset with {len(df)} rows, {len(df.columns)} columns"])
        
        # Basic data analysis
        analysis = {
            'summary_stats': df.describe() if not df.empty else None,
            'column_info': {col: str(df[col].dtype) for col in df.columns} if not df.empty else {},
            'missing_values': df.isnull().sum().to_dict() if not df.empty else {},
            'unique_values': {col: df[col].nunique() for col in df.columns if df[col].dtype == 'object'} if not df.empty else {}
        }
        
        return analysis
    
    def generate_response(self, query: str, data_analysis: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate AI response based on data analysis"""
        
        # Add audit step
        self.audit.add_step("AI_GENERATION", "Generating AI response", 
                           ["Data analysis results", "User query"])
        
        try:
            # Create context from data
            context = self._create_context(data_analysis, df)
            
            # Create prompt
            prompt = self._create_prompt(query, context)
            
            # Generate response using OpenAI v1.0+
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a government data analyst AI. Provide accurate, factual responses based only on the provided data. Always cite specific data points and be transparent about limitations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            ai_response = response.choices[0].message.content
            
            # Calculate trust score
            data_coverage = self._calculate_data_coverage(query, df)
            response_specificity = self._calculate_response_specificity(ai_response)
            source_quality = self._calculate_source_quality(df)
            
            trust_score = self.trust_calculator.calculate_trust_score(
                query, data_coverage, response_specificity, source_quality
            )
            
            # Add audit step
            self.audit.add_step("TRUST_CALCULATION", f"Trust score calculated: {trust_score['score']}", 
                               ["AI response", "Data quality metrics"])
            
            return {
                'response': ai_response,
                'trust_score': trust_score,
                'audit_trail': self.audit.get_trail(),
                'data_sources_used': list(df['source_dataset'].unique()) if 'source_dataset' in df.columns else ['Combined Dataset']
            }
            
        except Exception as e:
            # Fixed trust score structure for error cases
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
                'data_sources_used': []
            }
    
    def _create_context(self, analysis: Dict, df: pd.DataFrame) -> str:
        """Create context string from data analysis"""
        context_parts = []
        
        if not df.empty:
            context_parts.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")
            context_parts.append(f"Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
            
            # Add sample data
            sample_data = df.head(3).to_string()
            context_parts.append(f"Sample data:\n{sample_data}")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for OpenAI"""
        return f"""
        Based on the following government dataset information, please answer the user's question accurately and concisely.
        
        Data Context:
        {context}
        
        User Question: {query}
        
        Guidelines:
        - Only use information from the provided dataset
        - Be specific and cite data points where possible
        - If the data doesn't contain information to answer the question, say so clearly
        - Focus on accuracy over completeness
        - Highlight any limitations or caveats
        """
    
    def _calculate_data_coverage(self, query: str, df: pd.DataFrame) -> float:
        """Calculate how well the data covers the query"""
        if df.empty:
            return 0.0
        
        # Simple heuristic based on query keywords present in columns
        query_words = set(query.lower().split())
        column_words = set(' '.join(df.columns).lower().split())
        
        overlap = len(query_words.intersection(column_words))
        coverage = min(overlap / len(query_words) if query_words else 0, 1.0)
        return max(coverage, 0.5)  # Minimum 50% if data exists
    
    def _calculate_response_specificity(self, response: str) -> float:
        """Calculate response specificity"""
        # Simple heuristic based on response length and presence of numbers
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
        
        # Calculate based on completeness and consistency
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        quality = completeness
        
        return max(quality, 0.7)

# Main Streamlit App
def main():
    st.title("🏛️ Government Data Conversational AI")
    st.markdown("### Accurate and Trustworthy Chatbot for Data Interactions")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Show API status in sidebar
    st.sidebar.success("🔑 OpenAI API: Connected")
    
    # # Add some useful info in sidebar
    # st.sidebar.info("📋 The AI will automatically detect connections between your datasets and generate relevant questions.")
    # st.sidebar.info("📊 ProfileReports are generated automatically for data visualization and analysis.")
    
    # Data upload section
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
        
        # Check if files have changed
        current_files_hash = processor.get_files_hash(uploaded_files)
        files_changed = current_files_hash != st.session_state.uploaded_files_hash
        
        if files_changed:
            # Reset questions and profile reports when files change
            st.session_state.suggested_questions = []
            st.session_state.dataset_questions = {}
            st.session_state.profile_reports = {}
            st.session_state.uploaded_files_hash = current_files_hash
        
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            df, metadata = processor.load_file(uploaded_file)
            if df is not None:
                datasets[uploaded_file.name] = df
                st.session_state.data_sources[uploaded_file.name] = metadata
        
        if datasets:
            # Combine datasets
            combined_df = processor.combine_datasets(datasets)
            
            # Display data overview
            st.metric("📁 Data Sources", len(datasets))
            
            # Data Sources Preview with ProfileReport
            st.header("📋 Data Sources & Analysis")
            
            # Create tabs or expandable sections for each data source
            if len(datasets) == 1:
                # Single dataset - show directly
                dataset_name = list(datasets.keys())[0]
                df = datasets[dataset_name]
                metadata = st.session_state.data_sources[dataset_name]
                
                with st.expander(f"📄 {dataset_name}", expanded=True):
                    # Dataset metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Records", metadata['rows'])
                    with col2:
                        st.metric("Columns", len(metadata['columns']))
                    with col3:
                        file_size = f"{metadata['size'] / 1024:.1f} KB" if metadata['size'] < 1024*1024 else f"{metadata['size'] / (1024*1024):.1f} MB"
                        st.metric("Size", file_size)
                    
                    # Tabs for different views
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
                        
                        # Display profile report if it exists
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
                # Multiple datasets - show as expandable sections
                for dataset_name, df in datasets.items():
                    metadata = st.session_state.data_sources[dataset_name]
                    
                    with st.expander(f"📄 {dataset_name}"):
                        # Dataset metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Records", metadata['rows'])
                        with col2:
                            st.metric("Columns", len(metadata['columns']))
                        with col3:
                            file_size = f"{metadata['size'] / 1024:.1f} KB" if metadata['size'] < 1024*1024 else f"{metadata['size'] / (1024*1024):.1f} MB"
                            st.metric("Size", file_size)
                        
                        # Tabs for different views
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
                            
                            # Display profile report if it exists
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
            
            # Generate AI-powered question suggestions
            if not st.session_state.suggested_questions:
                with st.spinner("🤖 AI is analyzing your datasets and their connections to generate relevant questions..."):
                    try:
                        question_generator = QuestionGenerator(OPENAI_API_KEY)
                        
                        # Generate questions for all datasets
                        all_questions = question_generator.generate_questions_for_datasets(datasets)
                        st.session_state.dataset_questions = all_questions
                        
                        # Select top 5 questions
                        selected_questions = question_generator.select_top_questions(all_questions, 5)
                        st.session_state.suggested_questions = selected_questions
                        
                    except Exception as e:
                        st.warning(f"Unable to generate AI questions: {str(e)}")
                        # Use fallback questions
                        st.session_state.suggested_questions = [
                            "What are the main patterns across all datasets?",
                            "Are there any correlations between different data sources?",
                            "What outliers need attention?",
                            "How do the datasets complement each other?",
                            "What insights emerge from combining these datasets?"
                        ]
            
            # Question scaffolding
            st.header("❓ Question Guidance")
            
            if st.session_state.suggested_questions:
                st.markdown("**🤖 AI-Generated Questions Based on Your Data:**")
                
                # Show information about question generation
                if len(datasets) > 1:
                    st.info(f"📊 These questions were generated by analyzing {len(datasets)} datasets and their connections. Cross-dataset questions are prioritized to help you find relationships between different data sources.")
                else:
                    st.info("📊 These questions were automatically generated by analyzing your uploaded dataset.")
                
                for i, question in enumerate(st.session_state.suggested_questions):
                    if st.button(f"📝 {question}", key=f"ai_question_{i}"):
                        st.session_state.user_question = question
            
            # Show detailed question breakdown in expander
            if st.session_state.dataset_questions:
                with st.expander("🔍 View All Generated Questions by Source", expanded=False):
                    for source, questions in st.session_state.dataset_questions.items():
                        if questions:
                            if source == 'cross_dataset':
                                st.markdown(f"**🔗 Cross-Dataset Analysis Questions:**")
                            else:
                                st.markdown(f"**📄 Questions for {source}:**")
                            for j, q in enumerate(questions, 1):
                                st.write(f"{j}. {q}")
                            st.markdown("---")
            
            # Reset questions button
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
            
            # Query input
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
                        # Initialize AI
                        ai = ConversationalAI(OPENAI_API_KEY)
                        
                        # Analyze data
                        analysis = ai.analyze_data(combined_df, user_question)
                        
                        # Generate response
                        result = ai.generate_response(user_question, analysis, combined_df)
                        
                        # Display results
                        st.header("🤖 AI Response")
                        
                        # Trust score display
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
                        
                        # Trust score breakdown
                        with st.expander("🔍 Trust Score Breakdown"):
                            st.write("**Components:**")
                            if 'components' in trust_info:
                                for component, value in trust_info['components'].items():
                                    st.write(f"- {component.replace('_', ' ').title()}: {value}")
                            else:
                                st.write("Trust score components not available")
                        
                        # Audit trail
                        st.header("📋 Audit Trail")
                        audit_trail = result['audit_trail']
                        
                        for i, step in enumerate(audit_trail):
                            with st.expander(f"Step {i+1}: {step['step_type']}", expanded=False):
                                st.write(f"**Description:** {step['description']}")
                                st.write(f"**Timestamp:** {step['timestamp']}")
                                st.write(f"**Data Used:** {', '.join(step['data_used'])}")
                                st.write(f"**Step ID:** {step['step_id']}")
                        
                        # Data sources used
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
                        
                        # Save to conversation history
                        st.session_state.conversation_history.append({
                            'question': user_question,
                            'response': result['response'],
                            'trust_score': trust_info['score'],
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.write("Please check your API key configuration.")
            
            # Conversation history
            if st.session_state.conversation_history:
                st.header("📜 Conversation History")
                for i, conv in enumerate(reversed(st.session_state.conversation_history)):
                    with st.expander(f"Q{len(st.session_state.conversation_history)-i}: {conv['question'][:100]}..."):
                        st.write(f"**Question:** {conv['question']}")
                        st.write(f"**Response:** {conv['response']}")
                        st.write(f"**Trust Score:** {conv['trust_score']}")
                        st.write(f"**Time:** {conv['timestamp']}")
    
    else:
        st.info("👆 Please upload your government datasets to begin analysis.")
        
        # Show example datasets structure
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
