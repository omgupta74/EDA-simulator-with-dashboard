import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tkinter import Tk, filedialog, Button, Label, Frame, Canvas, Scrollbar, StringVar, Toplevel, BooleanVar
from tkinter.ttk import OptionMenu, Separator, Checkbutton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.stats import skew, normaltest
from tqdm import tqdm
import traceback
import warnings
from pandastable import Table
from sklearn.preprocessing import MinMaxScaler
import openpyxl


warnings.filterwarnings('ignore')


sns.set_style("whitegrid")
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

class DataLoader:
    """Enhanced data loader with progress tracking, smart dtype detection, and large dataset support"""
    def __init__(self):
        self.df = None
        self.file_path = None
        
    def load_dataset(self):
        """Load dataset with progress tracking and error handling"""
        root = Tk()
        root.withdraw()
        self.file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("Parquet files", "*.parquet"),
                ("All files", "*.*")
            ]
        )
        if not self.file_path:
            raise ValueError("No file selected.")
            
        print(f"Loading {self.file_path}...")
        
        try:
            if self.file_path.endswith('.csv'):
                chunk_size = 100000
                chunks = pd.read_csv(self.file_path, chunksize=chunk_size, low_memory=False)
                self.df = pd.concat(tqdm(chunks, desc="Loading CSV chunks"), ignore_index=True)
            elif self.file_path.endswith('.xlsx'):
                # Estimate total rows for chunking
                wb = openpyxl.load_workbook(self.file_path, read_only=True)
                ws = wb.active
                total_rows = ws.max_row - 1  # Exclude header
                wb.close()
                
                if total_rows > 1000000:
                    print("Warning: Large Excel file detected (>1M rows). This may take significant time.")
                
                chunk_size = 100000
                chunks = []
                for start_row in tqdm(range(0, total_rows, chunk_size), desc="Loading Excel chunks"):
                    chunk = pd.read_excel(
                        self.file_path,
                        engine='openpyxl',
                        skiprows=start_row,
                        nrows=chunk_size,
                        header=0 if start_row == 0 else None
                    )
                    chunks.append(chunk)
                self.df = pd.concat(chunks, ignore_index=True)
            elif self.file_path.endswith('.parquet'):
                self.df = pd.read_parquet(self.file_path)
                
            print(f"Successfully loaded {len(self.df)} rows.")
            print(f"Initial memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            return self.df
            
        except MemoryError:
            raise MemoryError("File too large for available memory. Try increasing chunk_size, using Parquet format, or using a machine with more RAM.")
        except Exception as e:
            raise Exception(f"Failed to load file: {str(e)}")
    
    def preprocess_data(self, handle_missing=True, remove_duplicates=True, handle_outliers=True):
        """Smart preprocessing pipeline with memory optimization"""
        if self.df is None:
            raise ValueError("No data loaded.")
            
    
        date_pattern = r'(date|time|year|month|day|timestamp|dt)'
        date_cols = [col for col in self.df.columns 
                    if re.search(date_pattern, col, re.IGNORECASE)]
        
        for col in self.df.columns:
            if col not in date_cols and self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col], errors='raise')
                    date_cols.append(col)
                except:
                    pass
        
        for col in date_cols:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
        # Optimize data types
        for col in self.df.select_dtypes(include=['float64']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')
        for col in self.df.select_dtypes(include=['int64']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
        for col in self.df.select_dtypes(include=['object']).columns:
            if self.df[col].nunique() / len(self.df) < 0.1:  # Less than 10% unique values
                self.df[col] = self.df[col].astype('category')
        
        # Handle missing values
        if handle_missing:
            self.df = self.df.apply(lambda col: col.fillna(col.mean()) 
                                  if np.issubdtype(col.dtype, np.number) 
                                  else col.fillna(col.mode()[0]) if not col.mode().empty else col)
        
    
        if remove_duplicates:
            self.df = self.df.drop_duplicates()
        
        # Handle outliers
        if handle_outliers:
            num_cols = self.df.select_dtypes(include=np.number).columns
            if not num_cols.empty:
                self.df[num_cols] = self.df[num_cols].apply(
                    lambda x: x.clip(*x.quantile([0.05, 0.95])))
        
        print(f"Memory usage after preprocessing: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return self.df


        
    def comprehensive_summary(self):
        """Generate detailed data profile"""
        summary = pd.DataFrame({
            'dtype': self.df.dtypes,
            'missing': self.df.isna().sum(),
            'unique': self.df.nunique(),
            'mean': self.df.select_dtypes(include=np.number).mean(),
            'median': self.df.select_dtypes(include=np.number).median(),
            'skewness': self.df.select_dtypes(include=np.number).apply(skew)
        })
        return summary
    
    def normality_tests(self):
        """Perform normality tests on numeric columns"""
        results = {}
        num_cols = self.df.select_dtypes(include=np.number).columns
        for col in num_cols:
            stat, p = normaltest(self.df[col].dropna())
            results[col] = {'statistic': stat, 'p-value': p}
        return pd.DataFrame(results).T
    
    def generate_insights(self):
        """Generate actionable insights from the data"""
        insights = []
        insights.append(f"📊 Dataset Contains: {len(self.df)} rows, {len(self.df.columns)} columns")
        
        # Missing values
        missing = self.df.isna().sum().sum()
        if missing > 0:
            missing_cols = self.df.columns[self.df.isna().any()].tolist()
            insights.append(f"⚠️ Missing Values: {missing} missing values in {len(missing_cols)} columns")
        
        # Skewness
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        skewed_cols = [(col, skew(self.df[col])) for col in numeric_cols if abs(skew(self.df[col])) > 1]
        if skewed_cols:
            insights.append("📈 Skewed Features:\n" + "\n".join([f"• {col}: {skew:.2f}" for col, skew in skewed_cols[:3]]))
        
        # Correlations
        corr_matrix = self.df.corr(numeric_only=True).abs()
        strong_corrs = corr_matrix.unstack().sort_values(ascending=False)
        strong_corrs = strong_corrs[strong_corrs.between(0.7, 0.99)]
        if not strong_corrs.empty:
            insights.append("🔗 Strong Correlations:\n" + "\n".join([f"• {cols[0]} ↔ {cols[1]}" for cols in strong_corrs.index[:3]]))
        
        return insights if insights else ["✅ No significant issues detected"]

class DataVisualizerGUI:
    """Interactive GUI dashboard with comprehensive visualizations"""
    def __init__(self, df, root):
        self.df = df
        self.root = root
        self.current_figures = []
        self.theme_mode = 'light'
        self.current_filters = {}
        
        # Configure main window
        self.root.title("Advanced EDA Dashboard")
        self.root.geometry("1366x768")
        self.root.state('zoomed')
        
        # Create UI components
        self._create_widgets()
        self._setup_columns_selector()
        self._create_status_bar()
        self._create_data_preview()
        self._create_theme_selector()
        
    def _create_widgets(self):
        """Initialize UI components with scrollable controls"""
        # Main container
        self.main_frame = Frame(self.root)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollable control panel
        self.control_canvas = Canvas(self.main_frame, width=300, bg='#f0f0f0')
        self.control_frame = Frame(self.control_canvas)
        self.control_scrollbar = Scrollbar(self.main_frame, orient="vertical", command=self.control_canvas.yview)
        
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        self.control_canvas.pack(side='left', fill='y')
        self.control_scrollbar.pack(side='left', fill='y')
        
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        self.control_frame.bind("<Configure>", lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))
        
        # Visualization area
        self.viz_frame = Frame(self.main_frame)
        self.viz_frame.pack(side='right', fill='both', expand=True)
        
        # Visualization canvas
        self.viz_canvas = Canvas(self.viz_frame)
        self.viz_scrollbar = Scrollbar(self.viz_frame, orient='vertical', command=self.viz_canvas.yview)
        self.scrollable_viz = Frame(self.viz_canvas)
        
        self.scrollable_viz.bind('<Configure>', lambda e: self.viz_canvas.configure(scrollregion=self.viz_canvas.bbox('all')))
        self.viz_canvas.create_window((0, 0), window=self.scrollable_viz, anchor='nw')
        self.viz_canvas.configure(yscrollcommand=self.viz_scrollbar.set)
        
        self.viz_canvas.pack(side='left', fill='both', expand=True)
        self.viz_scrollbar.pack(side='right', fill='y')
        
        # Visualization buttons
        buttons = [
            ('📁 Data Overview', self.show_data_overview),
            ('📊 Histogram', self.plot_histogram),
            ('📦 Box Plot', self.plot_boxplot),
            ('🖇 Scatter Plot', self.plot_scatter),
            ('📈 Line Graph', self.plot_line),
            ('📊 Bar Chart', self.plot_bar),
            ('🥧 Pie Chart', self.plot_pie),
            ('🔥 Heatmap', self.plot_heatmap),
            ('🔗 Correlation Matrix', self.plot_correlation),
            ('⏳ Time Series', self.plot_time_series),
            ('💡 Generate Insights', self.show_insights),
            ('📤 Export Report', self.export_report)
        ]
        
        for idx, (text, cmd) in enumerate(buttons):
            btn = Button(self.control_frame, text=text, command=cmd, 
                        width=20, bg='#4CAF50', fg='white', font=('Arial', 10))
            btn.grid(row=idx, column=0, pady=5, padx=5, sticky='ew')
            
    def _setup_columns_selector(self):
        """Column selection dropdowns"""
        Label(self.control_frame, text="Select Columns:", bg='#f0f0f0').grid(row=20, column=0, pady=10)
        
        self.selected_x = StringVar()
        self.selected_y = StringVar()
        
        self.x_dropdown = OptionMenu(self.control_frame, self.selected_x, *self.df.columns)
        self.x_dropdown.grid(row=21, column=0, pady=5, sticky='ew')
        
        self.y_dropdown = OptionMenu(self.control_frame, self.selected_y, *self.df.columns)
        self.y_dropdown.grid(row=22, column=0, pady=5, sticky='ew')
        
        # Add filter controls
        filter_frame = Frame(self.control_frame, bg='#f0f0f0')
        filter_frame.grid(row=23, column=0, pady=10, sticky='ew')
        
        Label(filter_frame, text="Filters:", bg='#f0f0f0').pack(side='top', anchor='w')
        self.filter_var = StringVar()
        self.filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var)
        self.filter_entry.pack(side='left', fill='x', expand=True)
        ttk.Button(filter_frame, text="Apply", command=self.apply_filter).pack(side='left')
        
    def _create_status_bar(self):
        """Create status bar at bottom"""
        self.status = StringVar()
        self.status.set("Ready")
        status_bar = Label(self.root, textvariable=self.status, bd=1, relief='sunken', anchor='w')
        status_bar.pack(side='bottom', fill='x')
        
    def _create_data_preview(self):
        """Add data preview button"""
        btn = Button(self.control_frame, text="🔍 Data Preview", command=self.show_data_preview,
                    width=20, bg='#2196F3', fg='white')
        btn.grid(row=24, column=0, pady=10, sticky='ew')
        
    def _create_theme_selector(self):
        """Add theme selector"""
        theme_frame = Frame(self.control_frame, bg='#f0f0f0')
        theme_frame.grid(row=25, column=0, pady=10, sticky='ew')
        
        Label(theme_frame, text="Theme:", bg='#f0f0f0').pack(side='left')
        self.theme_var = StringVar(value='light')
        ttk.OptionMenu(theme_frame, self.theme_var, 'Light', 'Light', 'Dark', 
                      command=self.change_theme).pack(side='right')
        
    def change_theme(self, theme):
        """Change application theme"""
        self.theme_mode = theme.lower()
        bg_color = '#333333' if self.theme_mode == 'dark' else '#f0f0f0'
        fg_color = 'white' if self.theme_mode == 'dark' else 'black'
        
        # Update main background
        self.root.config(bg=bg_color)
        self.main_frame.config(bg=bg_color)
        self.control_canvas.config(bg=bg_color)
        self.control_frame.config(bg=bg_color)
        self.viz_frame.config(bg=bg_color)
        
        # Update widget colors
        for widget in self.control_frame.winfo_children():
            if isinstance(widget, (Label, Frame)):
                widget.config(bg=bg_color, fg=fg_color)
                
    def apply_filter(self):
        """Apply pandas query filter"""
        try:
            query = self.filter_var.get()
            if query:
                self.df = self.df.query(query)
                self.status.set(f"Applied filter: {query}")
                self._update_column_dropdowns()
        except Exception as e:
            self.status.set(f"Filter error: {str(e)}")
            
    def _update_column_dropdowns(self):
        """Update dropdown menus after filtering"""
        cols = self.df.columns.tolist()
        self.x_dropdown['menu'].delete(0, 'end')
        self.y_dropdown['menu'].delete(0, 'end')
        
        for col in cols:
            self.x_dropdown['menu'].add_command(label=col, command=lambda v=col: self.selected_x.set(v))
            self.y_dropdown['menu'].add_command(label=col, command=lambda v=col: self.selected_y.set(v))
            
    def show_data_preview(self):
        """Show interactive data preview"""
        preview = Toplevel(self.root)
        preview.title("Data Preview")
        pt = Table(preview, dataframe=self.df.head(100))
        pt.show()
        
    def _clear_viz_area(self):
        """Clear previous visualizations"""
        for widget in self.scrollable_viz.winfo_children():
            widget.destroy()
        self.current_figures = []
        
    def _add_figure(self, fig, analysis_text=None):
        """Add visualization with analysis"""
        frame = Frame(self.scrollable_viz)
        frame.pack(fill='x', pady=10)
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        
        # Add export button
        btn = Button(toolbar, text="💾 Export", command=lambda: self.export_figure(fig))
        btn.pack(side='right')
        
        if analysis_text:
            analysis = Label(frame, text=analysis_text, wraplength=1000,
                            justify='left', bg='white', padx=10, pady=5)
            analysis.pack(fill='x', side='bottom')
        
        self.current_figures.append(canvas)

    def export_figure(self, fig):
        """Export figure to file"""
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All Files", "*.*")]
        )
        if path:
            if path.endswith('.pdf'):
                fig.savefig(path, format='pdf', bbox_inches='tight')
            else:
                fig.savefig(path, bbox_inches='tight')
            self.status.set(f"Exported figure to {path}")

    def show_data_overview(self):
        self._clear_viz_area()
        fig = Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        summary = DataAnalyzer(self.df).comprehensive_summary()
        table = ax.table(cellText=summary.values, colLabels=summary.columns,
                        rowLabels=summary.index, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        fig.suptitle('Dataset Overview', fontsize=14)
        
        analysis = "Dataset Summary:\n- Shows basic statistics\n- Check data types & missing values"
        self._add_figure(fig, analysis)

    def plot_histogram(self):
        col = self.selected_x.get()
        if not col: return
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        sns.histplot(self.df[col], kde=True, ax=ax, color='skyblue')
        ax.set_title(f'Histogram of {col}')
        
        stats = self.df[col].describe()
        analysis = (f"Histogram Analysis:\nMean: {stats['mean']:.2f}, Std: {stats['std']:.2f}\n"
                   "Shows data distribution and outliers")
        self._add_figure(fig, analysis)

    def plot_boxplot(self):
        col = self.selected_x.get()
        if not col: return
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        sns.boxplot(x=self.df[col], ax=ax, color='lightgreen')
        ax.set_title(f'Box Plot of {col}')
        
        q1, q3 = self.df[col].quantile([0.25, 0.75])
        analysis = (f"Box Plot Analysis:\nMedian: {self.df[col].median():.2f}\n"
                   f"IQR: {q3-q1:.2f} (Q1: {q1:.2f}, Q3: {q3:.2f})\n"
                   "Shows spread and outliers")
        self._add_figure(fig, analysis)

    def plot_scatter(self):
        x, y = self.selected_x.get(), self.selected_y.get()
        if not x or not y: return
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        sns.scatterplot(x=self.df[x], y=self.df[y], ax=ax, color='coral')
        ax.set_title(f'{x} vs {y} Scatter Plot')
        
        corr = self.df[[x, y]].corr().iloc[0,1]
        analysis = (f"Scatter Plot Analysis:\nCorrelation: {corr:.2f}\n"
                   "Positive trend: Points rise right\nNegative trend: Points fall right")
        self._add_figure(fig, analysis)

    def plot_line(self):
        x, y = self.selected_x.get(), self.selected_y.get()
        if not x or not y: return
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        sns.lineplot(x=self.df[x], y=self.df[y], ax=ax, color='purple')
        ax.set_title(f'{x} vs {y} Line Graph')
        
        analysis = "Line Graph Analysis:\nShows trends and patterns over continuous variables"
        self._add_figure(fig, analysis)

    def plot_bar(self):
        col = self.selected_x.get()
        if not col: return
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        if self.df[col].dtype == 'object':
            sns.countplot(x=self.df[col], ax=ax, palette='viridis')
            analysis = "Categorical Distribution\nShows frequency of each category"
        else:
            sns.barplot(x=self.df.index, y=self.df[col], ax=ax, ci=None)
            analysis = "Value Distribution\nShows magnitude of numerical values"
            
        ax.set_title(f'Bar Chart of {col}')
        ax.tick_params(axis='x', rotation=45)
        self._add_figure(fig, analysis)

    def plot_pie(self):
        col = self.selected_x.get()
        if not col: 
            return
        
        if self.df[col].nunique() > 20:
            self.status.set("Too many categories for pie chart (max 20)")
            return
            
        counts = self.df[col].value_counts()
        if len(counts) == 0:
            self.status.set("No data to plot")
            return

        self._clear_viz_area()
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        labels = [str(label)[:15] + '...' if len(str(label)) > 15 else str(label) 
                for label in counts.index]
                
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=labels,
            autopct=lambda p: f'{p:.1f}%' if p >= 1 else '',
            startangle=90,
            pctdistance=0.85,
            textprops={'fontsize': 8}
        )
        
        plt.setp(autotexts, size=8, weight="bold", color='white')
        ax.axis('equal')
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        fig.suptitle(f'Pie Chart of {col}', fontsize=14)
        
        ax.legend(
            wedges,
            labels,
            title="Categories",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=8
        )
        
        analysis = (f"Pie Chart Analysis:\nTotal Categories: {len(counts)}\n"
                   f"Dominant Category: {counts.idxmax()} ({counts.max()} entries)\n"
                   "Note: Percentages <1% are omitted for clarity")
        self._add_figure(fig, analysis)

    def plot_heatmap(self):
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        numeric_df = self.df.select_dtypes(include=np.number)
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        
        analysis = ("Heatmap Analysis:\n"
                   "Shows pairwise correlations between numerical features\n"
                   "Red: Positive, Blue: Negative correlation")
        self._add_figure(fig, analysis)

    def plot_correlation(self):
        self.plot_heatmap()

    def plot_time_series(self):
        date_col = self.selected_x.get()
        value_col = self.selected_y.get()
        
        if not date_col or not value_col:
            self.status.set("Please select both X (date) and Y (value) columns")
            return
            
        if not pd.api.types.is_numeric_dtype(self.df[value_col]):
            self.status.set(f"Error: {value_col} must be a numeric column")
            return
            
        try:
            temp_dates = pd.to_datetime(self.df[date_col], errors='coerce')
            if temp_dates.isna().all():
                self.status.set(f"Could not convert {date_col} to datetime")
                return
        except Exception as e:
            self.status.set(f"Date conversion error: {str(e)}")
            return

        self._clear_viz_area()
        fig = Figure(figsize=(12, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        try:
            sns.lineplot(x=temp_dates, y=self.df[value_col], ax=ax, errorbar=None)
            ax.set_title(f'{value_col} Time Series')
            fig.autofmt_xdate()
            
            z = np.polyfit(temp_dates.astype('int64'), self.df[value_col], 1)
            p = np.poly1d(z)
            ax.plot(temp_dates, p(temp_dates.astype('int64')), "r--")
            
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(self.df[[value_col]])
            anomalies = self.df[(scaled_values > 1.2) | (scaled_values < -0.2)]
            ax.scatter(temp_dates[anomalies.index], anomalies[value_col], 
                      color='red', label='Anomalies')
            
            analysis = (f"Time Series Analysis:\n"
                       f"Date Range: {temp_dates.min().date()} to {temp_dates.max().date()}\n"
                       f"Average {value_col}: {self.df[value_col].mean():.2f}\n"
                       f"Detected {len(anomalies)} anomalies")
        except Exception as e:
            self.status.set(f"Plotting error: {str(e)}")
            return
            
        self._add_figure(fig, analysis)
        self.status.set("Time series plot generated")

    def show_insights(self):
        self._clear_viz_area()
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        insights = DataAnalyzer(self.df).generate_insights()
        text = "🔍 Key Insights:\n\n" + "\n\n".join(insights)
        
        ax.text(0.05, 0.95, text, ha='left', va='top', transform=ax.transAxes,
               fontsize=12, wrap=True, bbox=dict(facecolor='white', alpha=0.8))
        
        fig.suptitle('Actionable Insights', fontsize=14)
        self._add_figure(fig)

    def export_report(self):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        analyzer = DataAnalyzer(self.df)
        pdf.multi_cell(0, 10, txt=f"Data Summary:\n{analyzer.comprehensive_summary().to_string()}")
        
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Key Insights:', 0, 1)
        pdf.set_font('Arial', '', 12)
        for insight in analyzer.generate_insights():
            pdf.multi_cell(0, 8, txt=f"• {insight}")
            
        for idx, fig in enumerate(self.current_figures):
            path = f"temp_fig_{idx}.png"
            fig.figure.savefig(path)
            pdf.add_page()
            pdf.image(path, x=10, y=10, w=180)
            pdf.ln(100)
            
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf")
        if save_path:
            pdf.output(save_path)
            self.status.set(f"Report exported to {save_path}")
            
def main():
    root = Tk()
    loader = DataLoader()
    
    try:
        df = loader.load_dataset()
        
        preprocess_window = Toplevel(root)
        preprocess_window.title("Preprocessing Options")
        preprocess_window.grab_set()

        handle_missing = BooleanVar(value=True)
        remove_duplicates = BooleanVar(value=True)
        handle_outliers = BooleanVar(value=True)

        Label(preprocess_window, text="Data Preprocessing Configuration", font=('Arial', 12, 'bold')).pack(pady=10)
        
        Checkbutton(preprocess_window, text="Handle Missing Values", variable=handle_missing).pack(anchor='w', padx=20)
        Checkbutton(preprocess_window, text="Remove Duplicates", variable=remove_duplicates).pack(anchor='w', padx=20)
        Checkbutton(preprocess_window, text="Handle Outliers", variable=handle_outliers).pack(anchor='w', padx=20)

        def apply_preprocessing():
            try:
                loader.preprocess_data(
                    handle_missing=handle_missing.get(),
                    remove_duplicates=remove_duplicates.get(),
                    handle_outliers=handle_outliers.get()
                )
                preprocess_window.destroy()
                DataVisualizerGUI(loader.df, root)
                root.mainloop()
            except Exception as e:
                print(f"Preprocessing error: {str(e)}")

        Button(preprocess_window, text="Apply Preprocessing", command=apply_preprocessing, 
              bg='#4CAF50', fg='white').pack(pady=20)

    except Exception as e:
        print(f"Error: {str(e)}")
        root.destroy()

if __name__ == "__main__":
    main()
