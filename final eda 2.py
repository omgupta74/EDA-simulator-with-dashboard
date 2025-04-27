import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tkinter import Tk, filedialog, Button, Label, Frame, Canvas, Scrollbar, StringVar, Toplevel, BooleanVar, ttk, Checkbutton, OptionMenu, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.stats import skew, normaltest
from tqdm import tqdm
import traceback
import warnings
from pandastable import Table
from sklearn.preprocessing import MinMaxScaler
import openpyxl
import xlrd
import gc
import os
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

class DataLoader:
    def __init__(self):
        self.df = None
        self.file_path = None
        
    def load_dataset(self):
        root = Tk()
        root.withdraw()
        self.file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
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
                engine = 'openpyxl'
                header_df = pd.read_excel(self.file_path, engine=engine, nrows=0)
                columns = header_df.columns
                
                chunk_size = 100000
                chunks = []
                for chunk in tqdm(pd.read_excel(self.file_path, engine=engine, chunksize=chunk_size, 
                                             header=0, index_col=None),
                                desc="Loading Excel (.xlsx) chunks"):
                    chunk.columns = columns
                    chunk = chunk.dropna(how='all')
                    if not chunk.empty:
                        chunks.append(chunk)
                    del chunk
                    gc.collect()
                
                if not chunks:
                    raise ValueError("No valid data found in .xlsx file.")
                
                self.df = pd.concat(chunks, ignore_index=True)
                del chunks
                gc.collect()
                
            elif self.file_path.endswith('.xls'):
                try:
                    xl_file = pd.ExcelFile(self.file_path, engine='xlrd')
                    sheets = xl_file.sheet_names
                    print(f"Found sheets: {sheets}")
                    if not sheets:
                        raise ValueError("No sheets found in .xls file.")
                    
                    self.df = xl_file.parse(sheets[0], index_col=None)
                    print(f"Loaded sheet: {sheets[0]}")
                    
                    self.df = self.df.dropna(how='all')
                    
                    if self.df.empty:
                        raise ValueError(f"Sheet {sheets[0]} contains no valid data.")
                    
                    if len(sheets) > 1:
                        print(f"Warning: Only the first sheet ({sheets[0]}) was loaded. Other sheets: {sheets[1:]}")
                    
                except Exception as e:
                    raise Exception(f"Failed to load .xls file: {str(e)}")
                
            elif self.file_path.endswith('.parquet'):
                self.df = pd.read_parquet(self.file_path)
            else:
                raise ValueError(f"Unsupported file format: {self.file_path}")
            
            if self.df is None:
                raise ValueError("Failed to load data: DataFrame is None.")
            if self.df.empty:
                raise ValueError("Loaded DataFrame is empty.")
                
            print(f"Successfully loaded {len(self.df)} rows and {len(self.df.columns)} columns.")
            print(f"Initial memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            return self.df
            
        except Exception as e:
            print(f"Detailed error: {traceback.format_exc()}")
            raise Exception(f"Failed to load file: {str(e)}")
    
    def preprocess_data(self, handle_missing=True, remove_duplicates=True, handle_outliers=True):
        if self.df is None:
            raise ValueError("No data loaded.")
            
        print("Starting preprocessing...")
        processed_df = self.df.copy()
        
        date_pattern = r'(date|time|year|month|day|timestamp|dt)'
        date_cols = [col for col in processed_df.columns 
                    if re.search(date_pattern, col, re.IGNORECASE)]
        
        for col in processed_df.columns:
            if col not in date_cols and processed_df[col].dtype == 'object':
                try:
                    temp_dates = pd.to_datetime(processed_df[col], errors='coerce')
                    if temp_dates.notna().sum() > len(processed_df) * 0.5:
                        date_cols.append(col)
                        processed_df[col] = temp_dates
                except Exception as e:
                    print(f"Date conversion failed for column {col}: {str(e)}")
        
        if handle_missing:
            print("Handling missing values...")
            for col in processed_df.columns:
                if processed_df[col].isna().sum() > 0:
                    print(f"Processing missing values in column: {col} (dtype: {processed_df[col].dtype})")
                    if np.issubdtype(processed_df[col].dtype, np.number):
                        median_val = processed_df[col].median()
                        if not np.isnan(median_val):
                            processed_df[col] = processed_df[col].fillna(median_val)
                        else:
                            processed_df[col] = processed_df[col].fillna(0)
                    elif processed_df[col].dtype.name == 'category':
                        mode_val = processed_df[col].mode()
                        if not mode_val.empty:
                            processed_df[col] = processed_df[col].cat.add_categories([mode_val[0]]).fillna(mode_val[0])
                        else:
                            new_category = 'Unknown'
                            processed_df[col] = processed_df[col].cat.add_categories([new_category]).fillna(new_category)
                    else:
                        mode_val = processed_df[col].mode()
                        if not mode_val.empty:
                            processed_df[col] = processed_df[col].fillna(mode_val[0])
                        else:
                            processed_df[col] = processed_df[col].fillna('Unknown')
        
        print("Optimizing data types...")
        for col in processed_df.select_dtypes(include=['float64']).columns:
            try:
                processed_df[col] = pd.to_numeric(processed_df[col], downcast='float')
            except Exception as e:
                print(f"Float downcast failed for column {col}: {str(e)}")
                
        for col in processed_df.select_dtypes(include=['int64']).columns:
            try:
                processed_df[col] = pd.to_numeric(processed_df[col], downcast='integer')
            except Exception as e:
                print(f"Integer downcast failed for column {col}: {str(e)}")
                
        for col in processed_df.select_dtypes(include=['object']).columns:
            if col not in date_cols and processed_df[col].nunique() / len(processed_df) < 0.1:
                try:
                    processed_df[col] = processed_df[col].astype('category')
                    print(f"Converted {col} to category (nunique: {processed_df[col].nunique()})")
                except Exception as e:
                    print(f"Category conversion failed for column {col}: {str(e)}")
        
        if remove_duplicates:
            initial_rows = len(processed_df)
            processed_df = processed_df.drop_duplicates()
            print(f"Removed {initial_rows - len(processed_df)} duplicate rows.")
        
        if handle_outliers:
            print("Handling outliers...")
            num_cols = processed_df.select_dtypes(include=np.number).columns
            if not num_cols.empty:
                for col in num_cols:
                    try:
                        q05, q95 = processed_df[col].quantile([0.05, 0.95])
                        processed_df[col] = processed_df[col].clip(lower=q05, upper=q95)
                    except Exception as e:
                        print(f"Outlier handling failed for column {col}: {str(e)}")
        
        self.df = processed_df
        print(f"Memory usage after preprocessing: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("Preprocessing completed successfully.")
        return self.df

class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        
    def comprehensive_summary(self):
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
        results = {}
        num_cols = self.df.select_dtypes(include=np.number).columns
        for col in num_cols:
            stat, p = normaltest(self.df[col].dropna())
            results[col] = {'statistic': stat, 'p-value': p}
        return pd.DataFrame(results).T
    
    def generate_insights(self):
        insights = []
        insights.append(f"📊 Dataset Contains: {len(self.df)} rows, {len(self.df.columns)} columns")
        
        missing = self.df.isna().sum().sum()
        if missing > 0:
            missing_cols = self.df.columns[self.df.isna().any()].tolist()
            insights.append(f"⚠️ Missing Values: {missing} missing values in {len(missing_cols)} columns")
        
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        skewed_cols = [(col, skew(self.df[col])) for col in numeric_cols if abs(skew(self.df[col])) > 1]
        if skewed_cols:
            insights.append("📈 Skewed Features:\n" + "\n".join([f"• {col}: {skew:.2f}" for col, skew in skewed_cols[:3]]))
        
        corr_matrix = self.df.corr(numeric_only=True).abs()
        strong_corrs = corr_matrix.unstack().sort_values(ascending=False)
        strong_corrs = strong_corrs[strong_corrs.between(0.7, 0.99)]
        if not strong_corrs.empty:
            insights.append("🔗 Strong Correlations:\n" + "\n".join([f"• {cols[0]} ↔ {cols[1]}" for cols in strong_corrs.index[:3]]))
        
        return insights if insights else ["✅ No significant issues detected"]

class DataVisualizerGUI:
    def __init__(self, df, root):
        self.df = df
        self.root = root
        self.current_figures = []
        self.theme_mode = 'light'
        self.current_filters = {}
        self.viz_mode = StringVar(value='2D')  # Default to 2D
        
        self.root.title("2D/3D EDA Dashboard with Explanations")
        self.root.state('zoomed')
        self.root.minsize(800, 600)
        
        self._create_widgets()
        self._setup_columns_selector()
        self._create_status_bar()
        self._create_data_preview()
        self._create_theme_selector()
        self._create_viz_mode_selector()
        self._set_default_selections()
        
    def _create_widgets(self):
        self.main_frame = Frame(self.root, bg='#f0f0f0')
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Left control panel
        self.control_canvas = Canvas(self.main_frame, width=250, bg='#2e2e2e', highlightthickness=0)
        self.control_frame = Frame(self.control_canvas, bg='#2e2e2e')
        self.control_scrollbar = Scrollbar(self.main_frame, orient="vertical", command=self.control_canvas.yview)
        
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        self.control_canvas.pack(side='left', fill='y', padx=(0, 5))
        self.control_scrollbar.pack(side='left', fill='y')
        
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        self.control_frame.bind("<Configure>", lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))
        
        # Right visualization area
        self.viz_frame = Frame(self.main_frame, bg='#f0f0f0', bd=1, relief='solid')
        self.viz_frame.pack(side='right', fill='both', expand=True)
        
        self.viz_canvas = Canvas(self.viz_frame, bg='#f0f0f0', highlightthickness=0)
        self.viz_scrollbar_y = Scrollbar(self.viz_frame, orient='vertical', command=self.viz_canvas.yview)
        self.viz_scrollbar_x = Scrollbar(self.viz_frame, orient='horizontal', command=self.viz_canvas.xview)
        self.scrollable_viz = Frame(self.viz_canvas, bg='#f0f0f0')
        
        self.viz_canvas.bind_all("<MouseWheel>", lambda event: self.viz_canvas.yview_scroll(int(-1*(event.delta/120)), "units"))
        self.viz_canvas.bind('<Configure>', lambda e: self.viz_canvas.configure(scrollregion=self.viz_canvas.bbox('all')))
        self.scrollable_viz.bind('<Configure>', lambda e: self.viz_canvas.configure(scrollregion=self.viz_canvas.bbox('all')))
        
        self.canvas_window = self.viz_canvas.create_window((0, 0), window=self.scrollable_viz, anchor='nw')
        self.viz_canvas.configure(yscrollcommand=self.viz_scrollbar_y.set, xscrollcommand=self.viz_scrollbar_x.set)
        
        self.viz_canvas.pack(side='top', fill='both', expand=True)
        self.viz_scrollbar_y.pack(side='right', fill='y')
        self.viz_scrollbar_x.pack(side='bottom', fill='x')
        
        def update_canvas_width(event):
            canvas_width = self.viz_canvas.winfo_width()
            self.viz_canvas.itemconfig(self.canvas_window, width=canvas_width)
        self.viz_canvas.bind('<Configure>', update_canvas_width, add='+')
        
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
                        width=20, bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                        relief='flat', pady=5)
            btn.grid(row=idx, column=0, pady=3, padx=10, sticky='ew')

    def _setup_columns_selector(self):
        Label(self.control_frame, text="Select Columns:", bg='#2e2e2e', fg='white', 
              font=('Arial', 10, 'bold')).grid(row=20, column=0, pady=(20, 5), padx=10, sticky='w')
        
        self.selected_x = StringVar()
        self.selected_y = StringVar()
        self.selected_z = StringVar()
        
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        all_cols = self.df.columns.tolist()
        
        self.x_dropdown = OptionMenu(self.control_frame, self.selected_x, *all_cols)
        self.y_dropdown = OptionMenu(self.control_frame, self.selected_y, *numeric_cols)
        self.z_dropdown = OptionMenu(self.control_frame, self.selected_z, *numeric_cols)
        
        self.x_dropdown.config(bg='#4CAF50', fg='white', font=('Arial', 10), width=18)
        self.y_dropdown.config(bg='#4CAF50', fg='white', font=('Arial', 10), width=18)
        self.z_dropdown.config(bg='#4CAF50', fg='white', font=('Arial', 10), width=18)
        
        self.x_dropdown.grid(row=21, column=0, pady=5, padx=10, sticky='ew')
        self.y_dropdown.grid(row=22, column=0, pady=5, padx=10, sticky='ew')
        self.z_dropdown.grid(row=23, column=0, pady=5, padx=10, sticky='ew')
        
        filter_frame = Frame(self.control_frame, bg='#2e2e2e')
        filter_frame.grid(row=24, column=0, pady=10, padx=10, sticky='ew')
        
        Label(filter_frame, text="Filters:", bg='#2e2e2e', fg='white', font=('Arial', 10, 'bold')).pack(side='top', anchor='w')
        self.filter_var = StringVar()
        self.filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, font=('Arial', 10))
        self.filter_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(filter_frame, text="Apply", command=self.apply_filter).pack(side='left')

    def _create_status_bar(self):
        self.status = StringVar()
        self.status.set("Ready")
        status_bar = Label(self.root, textvariable=self.status, bd=1, relief='sunken', anchor='w', 
                          font=('Arial', 10), bg='#e0e0e0')
        status_bar.pack(side='bottom', fill='x')

    def _create_data_preview(self):
        btn = Button(self.control_frame, text="🔍 Data Preview", command=self.show_data_preview,
                    width=20, bg='#2196F3', fg='white', font=('Arial', 10, 'bold'), 
                    relief='flat', pady=5)
        btn.grid(row=25, column=0, pady=10, padx=10, sticky='ew')

    def _create_theme_selector(self):
        theme_frame = Frame(self.control_frame, bg='#2e2e2e')
        theme_frame.grid(row=26, column=0, pady=10, padx=10, sticky='ew')
        
        Label(theme_frame, text="Theme:", bg='#2e2e2e', fg='white', font=('Arial', 10, 'bold')).pack(side='left')
        self.theme_var = StringVar(value='light')
        ttk.OptionMenu(theme_frame, self.theme_var, 'Light', 'Light', 'Dark', 
                      command=self.change_theme).pack(side='right')

    def _create_viz_mode_selector(self):
        mode_frame = Frame(self.control_frame, bg='#2e2e2e')
        mode_frame.grid(row=27, column=0, pady=10, padx=10, sticky='ew')
        
        Label(mode_frame, text="Viz Mode:", bg='#2e2e2e', fg='white', font=('Arial', 10, 'bold')).pack(side='left')
        ttk.OptionMenu(mode_frame, self.viz_mode, '2D', '2D', '3D', 
                      command=lambda m: self.status.set(f"Switched to {m} mode")).pack(side='right')

    def _set_default_selections(self):
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        
        if date_cols:
            self.selected_x.set(date_cols[0])
        if numeric_cols:
            self.selected_y.set(numeric_cols[0])
            if len(numeric_cols) > 1:
                self.selected_z.set(numeric_cols[1])

    def change_theme(self, theme):
        self.theme_mode = theme.lower()
        bg_color = '#333333' if self.theme_mode == 'dark' else '#f0f0f0'
        fg_color = 'white' if self.theme_mode == 'dark' else 'black'
        canvas_bg = '#444444' if self.theme_mode == 'dark' else '#f0f0f0'
        
        self.root.config(bg=bg_color)
        self.main_frame.config(bg=bg_color)
        self.control_canvas.config(bg=canvas_bg)
        self.control_frame.config(bg=canvas_bg)
        self.viz_frame.config(bg=bg_color)
        self.scrollable_viz.config(bg=bg_color)
        self.viz_canvas.config(bg=canvas_bg)
        
        for widget in self.control_frame.winfo_children():
            if isinstance(widget, (Label, Frame)):
                widget.config(bg=canvas_bg, fg=fg_color)
            elif isinstance(widget, Button):
                widget.config(bg='#4CAF50' if widget['text'] != '🔍 Data Preview' else '#2196F3', fg='white')
        
        for frame in self.scrollable_viz.winfo_children():
            frame.config(bg=bg_color)
            for child in frame.winfo_children():
                if isinstance(child, Label):
                    child.config(bg=bg_color, fg=fg_color)
                elif isinstance(child, Frame):
                    child.config(bg=bg_color)
                    for btn in child.winfo_children():
                        if isinstance(btn, Button):
                            btn.config(bg='#4CAF50', fg='white')

    def apply_filter(self):
        try:
            query = self.filter_var.get()
            if query:
                self.df = self.df.query(query)
                self.status.set(f"Applied filter: {query}")
                self._update_column_dropdowns()
        except Exception as e:
            messagebox.showerror("Filter Error", f"Failed to apply filter: {str(e)}")
            self.status.set("Filter application failed")

    def _update_column_dropdowns(self):
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        all_cols = self.df.columns.tolist()
        
        self.selected_x.set('')
        self.selected_y.set('')
        self.selected_z.set('')
        
        menu_x = self.x_dropdown['menu']
        menu_y = self.y_dropdown['menu']
        menu_z = self.z_dropdown['menu']
        
        menu_x.delete(0, 'end')
        menu_y.delete(0, 'end')
        menu_z.delete(0, 'end')
        
        for col in all_cols:
            menu_x.add_command(label=col, command=lambda v=col: self.selected_x.set(v))
        for col in numeric_cols:
            menu_y.add_command(label=col, command=lambda v=col: self.selected_y.set(v))
            menu_z.add_command(label=col, command=lambda v=col: self.selected_z.set(v))
        
        self._set_default_selections()

    def show_data_preview(self):
        preview = Toplevel(self.root)
        preview.title("Data Preview")
        pt = Table(preview, dataframe=self.df.head(100))
        pt.show()
        
    def _clear_viz_area(self):
        for widget in self.scrollable_viz.winfo_children():
            widget.destroy()
        self.current_figures = []
        
    def _add_figure(self, fig, analysis_text=None):
        frame = Frame(self.scrollable_viz, bg='#f0f0f0')
        frame.pack(fill='both', pady=10, padx=10, expand=True)
        
        canvas_width = self.viz_canvas.winfo_width()
        if canvas_width > 0:
            fig.set_size_inches(canvas_width / 100, 8)
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        
        btn = Button(toolbar, text="💾 Export", command=lambda: self.export_figure(fig), bg='#4CAF50', fg='white')
        btn.pack(side='right')
        
        if analysis_text:
            analysis = Label(frame, text=analysis_text, wraplength=canvas_width-20, justify='left', 
                            bg='#ffffff', padx=10, pady=5, font=('Arial', 10))
            analysis.pack(fill='x', side='bottom')
        
        self.current_figures.append(canvas)

    def export_figure(self, fig):
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
        
        analysis = ("Dataset Overview Explanation:\n"
                   "This table provides a summary of the dataset, including data types, missing values, unique entries, "
                   "and statistical measures (mean, median, skewness) for numerical columns. Use this to identify data "
                   "quality issues (e.g., high missing values) or skewed distributions that may require preprocessing.")
        self._add_figure(fig, analysis)

    def plot_histogram(self):
        x_col = self.selected_x.get()
        z_col = self.selected_z.get()
        if not x_col or x_col not in self.df.columns:
            self.status.set("Please select a valid X column")
            return
        
        if not pd.api.types.is_numeric_dtype(self.df[x_col]):
            self.status.set("X must be a numeric column")
            return

        self._clear_viz_area()
        fig = Figure(figsize=(10, 8), dpi=100)
        if self.viz_mode.get() == '3D' and z_col and z_col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[z_col]):
            ax = fig.add_subplot(111, projection='3d')
            hist, xedges, yedges = np.histogram2d(self.df[x_col].dropna(), self.df[z_col].dropna(), bins=20)
            xpos, ypos = np.meshgrid(xedges[:-1] + np.diff(xedges) / 2, yedges[:-1] + np.diff(yedges) / 2)
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros_like(xpos)
            dx = dy = np.diff(xedges)[0]
            dz = hist.flatten()
            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='skyblue', alpha=0.8)
            ax.set_xlabel(x_col)
            ax.set_ylabel('Frequency')
            ax.set_zlabel(z_col)
            ax.set_title(f'3D Histogram of {x_col} vs {z_col}', fontsize=14)
            analysis = (f"3D Histogram Explanation:\n"
                       f"This 3D histogram displays the distribution of {x_col} with {z_col} as the height (frequency). "
                       f"Each bar's height represents how often values occur. Rotate the plot to explore patterns. "
                       f"Key stats: Mean {x_col} = {self.df[x_col].mean():.2f}, Mean {z_col} = {self.df[z_col].mean():.2f}. "
                       "Look for peaks (high frequency areas) or gaps (missing data ranges).")
        else:
            ax = fig.add_subplot(111)
            sns.histplot(self.df[x_col], kde=True, ax=ax, color='skyblue')
            ax.set_title(f'Histogram of {x_col}')
            stats = self.df[x_col].describe()
            analysis = (f"Histogram Explanation:\n"
                       "This graph shows the frequency distribution of {x_col}. The x-axis represents value ranges, "
                       "and the y-axis shows how often each range occurs. The curve (KDE) estimates the data's smoothness. "
                       f"Key stats: Mean = {stats['mean']:.2f}, Std = {stats['std']:.2f}, Min = {stats['min']:.2f}, "
                       f"Max = {stats['max']:.2f}. A bell shape indicates normal distribution; skewness suggests outliers.")

        self._add_figure(fig, analysis)
        self.status.set(f"{self.viz_mode.get()} Histogram generated successfully")

    def plot_boxplot(self):
        x_col = self.selected_x.get()
        z_col = self.selected_z.get()
        if not x_col or x_col not in self.df.columns:
            self.status.set("Please select a valid X column")
            return
        
        if not pd.api.types.is_numeric_dtype(self.df[x_col]):
            self.status.set("X must be a numeric column")
            return

        self._clear_viz_area()
        fig = Figure(figsize=(10, 8), dpi=100)
        if self.viz_mode.get() == '3D' and z_col and z_col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[z_col]):
            ax = fig.add_subplot(111, projection='3d')
            data = self.df[[x_col, z_col]].dropna()
            q1 = data[x_col].quantile(0.25)
            q3 = data[x_col].quantile(0.75)
            median = data[x_col].median()
            iqr = q3 - q1
            lower_whisker = data[x_col].min()
            upper_whisker = data[x_col].max()
            ax.bar3d(0, 0, lower_whisker, 1, 1, iqr, color='lightgreen', alpha=0.8)
            ax.plot([0, 0], [0, 0], [median], 'r-', lw=2)
            ax.set_xlabel('Box')
            ax.set_ylabel('Depth')
            ax.set_zlabel(x_col)
            ax.set_title(f'3D Box Plot of {x_col} with {z_col} Depth', fontsize=14)
            analysis = (f"3D Box Plot Explanation:\n"
                       "This 3D box plot visualizes the spread of {x_col} with {z_col} as depth. The central red line "
                       f"is the median ({median:.2f}), the green box spans the IQR ({iqr:.2f} from Q1 {q1:.2f} to Q3 {q3:.2f}), "
                       "and whiskers extend to min/max values. Rotate to see depth influence. Outliers appear as points "
                       "outside whiskers if present.")
        else:
            ax = fig.add_subplot(111)
            sns.boxplot(x=self.df[x_col], ax=ax, color='lightgreen')
            ax.set_title(f'Box Plot of {x_col}')
            q1, q3 = self.df[x_col].quantile([0.25, 0.75])
            analysis = (f"Box Plot Explanation:\n"
                       "This box plot shows the distribution of {x_col}. The box covers the interquartile range (IQR) "
                       f"from Q1 ({q1:.2f}) to Q3 ({q3:.2f}), with the median line inside. Whiskers extend to the min/max "
                       f"values, and dots (if any) are outliers. IQR = {q3-q1:.2f}. Use this to spot variability and "
                       "outliers.")

        self._add_figure(fig, analysis)
        self.status.set(f"{self.viz_mode.get()} Box Plot generated successfully")

    def plot_scatter(self):
        x_col = self.selected_x.get()
        y_col = self.selected_y.get()
        z_col = self.selected_z.get()
        if not x_col or not y_col or x_col not in self.df.columns or y_col not in self.df.columns:
            self.status.set("Please select valid X and Y columns")
            return
        
        if not (pd.api.types.is_numeric_dtype(self.df[x_col]) and pd.api.types.is_numeric_dtype(self.df[y_col])):
            self.status.set("X and Y must be numeric columns")
            return

        self._clear_viz_area()
        fig = Figure(figsize=(10, 8), dpi=100)
        if self.viz_mode.get() == '3D' and z_col and z_col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[z_col]):
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.df[x_col], self.df[y_col], self.df[z_col], c='coral', alpha=0.6)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_zlabel(z_col)
            ax.set_title(f'3D Scatter Plot: {x_col} vs {y_col} vs {z_col}', fontsize=14)
            corr_xy = self.df[[x_col, y_col]].corr().iloc[0,1]
            corr_xz = self.df[[x_col, z_col]].corr().iloc[0,1]
            analysis = (f"3D Scatter Plot Explanation:\n"
                       "This 3D scatter plot shows the relationship between {x_col}, {y_col}, and {z_col}. Each point "
                       f"represents a data entry. Correlation X-Y = {corr_xy:.2f}, X-Z = {corr_xz:.2f}. A positive trend "
                       "means values increase together; negative means one decreases as the other increases. Rotate to "
                       "explore depth and clusters.")
        else:
            ax = fig.add_subplot(111)
            sns.scatterplot(x=self.df[x_col], y=self.df[y_col], ax=ax, color='coral')
            ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
            corr = self.df[[x_col, y_col]].corr().iloc[0,1]
            analysis = (f"Scatter Plot Explanation:\n"
                       "This scatter plot displays the relationship between {x_col} and {y_col}. Each point is a data "
                       f"entry. Correlation = {corr:.2f}. A positive trend (points rising right) suggests a direct "
                       "relationship; a negative trend (points falling right) suggests an inverse relationship. Look "
                       "for clusters or outliers.")

        self._add_figure(fig, analysis)
        self.status.set(f"{self.viz_mode.get()} Scatter Plot generated successfully")

    def plot_line(self):
        x_col = self.selected_x.get()
        y_col = self.selected_y.get()
        z_col = self.selected_z.get()
        if not x_col or not y_col or x_col not in self.df.columns or y_col not in self.df.columns:
            self.status.set("Please select valid X and Y columns")
            return
        
        if not (pd.api.types.is_numeric_dtype(self.df[x_col]) and pd.api.types.is_numeric_dtype(self.df[y_col])):
            self.status.set("X and Y must be numeric columns")
            return

        self._clear_viz_area()
        fig = Figure(figsize=(12, 8), dpi=100)
        if self.viz_mode.get() == '3D' and z_col and z_col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[z_col]):
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(self.df[x_col], self.df[y_col], self.df[z_col], color='purple', lw=1)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_zlabel(z_col)
            ax.set_title(f'3D Line Graph: {x_col} vs {y_col} vs {z_col}', fontsize=14)
            analysis = (f"3D Line Graph Explanation:\n"
                       "This 3D line graph tracks the trend of {y_col} and {z_col} over {x_col}. The line connects "
                       "sequential data points, showing changes over time or another variable. Rotate to see how {z_col} "
                       "adds depth to the trend. Steep slopes indicate rapid changes; flat lines suggest stability.")
        else:
            ax = fig.add_subplot(111)
            sns.lineplot(x=self.df[x_col], y=self.df[y_col], ax=ax, color='purple')
            ax.set_title(f'Line Graph: {x_col} vs {y_col}')
            analysis = (f"Line Graph Explanation:\n"
                       "This line graph shows the trend of {y_col} over {x_col}. The line connects data points to "
                       "reveal patterns or changes over time or another variable. Steep slopes indicate rapid changes, "
                       "while flat lines suggest stability. Look for peaks, troughs, or seasonal patterns.")

        self._add_figure(fig, analysis)
        self.status.set(f"{self.viz_mode.get()} Line Graph generated successfully")

    def plot_bar(self):
        x_col = self.selected_x.get()
        z_col = self.selected_z.get()
        if not x_col or x_col not in self.df.columns:
            self.status.set("Please select a valid X column")
            return
        
        self._clear_viz_area()
        fig = Figure(figsize=(10, 8), dpi=100)
        if self.viz_mode.get() == '3D' and z_col and z_col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[z_col]):
            ax = fig.add_subplot(111, projection='3d')
            data = self.df.groupby(x_col)[z_col].mean().reset_index()
            x_pos = np.arange(len(data))
            ax.bar3d(x_pos, np.zeros(len(data)), np.zeros(len(data)), 0.5, 0.5, data[z_col], color='b', alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(data[x_col], rotation=45)
            ax.set_xlabel(x_col)
            ax.set_ylabel('Depth')
            ax.set_zlabel(z_col)
            ax.set_title(f'3D Bar Chart of {x_col} vs {z_col}', fontsize=14)
            analysis = (f"3D Bar Chart Explanation:\n"
                       "This 3D bar chart compares the average {z_col} for each {x_col} category. The height of each "
                       "bar represents the mean value, with depth adding a visual layer. Rotate to see variations. "
                       "Tall bars indicate higher averages; compare bar heights to identify significant differences.")
        else:
            ax = fig.add_subplot(111)
            if self.df[x_col].dtype == 'object' or self.df[x_col].dtype.name == 'category':
                sns.countplot(x=self.df[x_col], ax=ax, palette='viridis')
                analysis = (f"Bar Chart Explanation:\n"
                           "This bar chart shows the frequency of each category in {x_col}. The height of each bar "
                           "represents the count of occurrences. Use this to identify dominant categories or "
                           "imbalances in the data.")
            else:
                sns.barplot(x=self.df.index, y=self.df[x_col], ax=ax, errorbar=None)
                analysis = (f"Bar Chart Explanation:\n"
                           "This bar chart displays the magnitude of {x_col} values. The height of each bar "
                           "represents the value at each index. Compare bar heights to see variations or trends.")
            ax.set_title(f'Bar Chart of {x_col}')
            ax.tick_params(axis='x', rotation=45)

        self._add_figure(fig, analysis)
        self.status.set(f"{self.viz_mode.get()} Bar Chart generated successfully")

    def plot_pie(self):
        col = self.selected_x.get()
        if not col or col not in self.df.columns:
            self.status.set("Please select a valid column for X")
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
        
        analysis = (f"Pie Chart Explanation:\n"
                   "This pie chart illustrates the proportional distribution of categories in {col}. Each slice's size "
                   f"represents its percentage (e.g., {counts.idxmax()} at {counts.max()/counts.sum()*100:.1f}%). "
                   "Slices <1% are omitted for clarity. Use this to see which categories dominate or if the data is "
                   "evenly distributed.")

        self._add_figure(fig, analysis)
        self.status.set("Pie Chart generated successfully")

    def plot_heatmap(self):
        self._clear_viz_area()
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        numeric_df = self.df.select_dtypes(include=np.number)
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        
        analysis = ("Heatmap Explanation:\n"
                   "This heatmap displays the correlation between numerical columns. Values range from -1 to 1: "
                   "1 (red) indicates a strong positive correlation, -1 (blue) a strong negative correlation, and 0 "
                   "(white) no correlation. The numbers inside cells are correlation coefficients. Use this to "
                   "identify related variables for further analysis.")

        self._add_figure(fig, analysis)
        self.status.set("Correlation Heatmap generated successfully")

    def plot_correlation(self):
        self.plot_heatmap()

    def plot_time_series(self):
        date_col = self.selected_x.get()
        value_col = self.selected_y.get()
        z_col = self.selected_z.get()
        
        if not date_col or not value_col or date_col not in self.df.columns or value_col not in self.df.columns:
            self.status.set("Please select valid X (date) and Y (value) columns")
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
        fig = Figure(figsize=(12, 8), dpi=100)
        if self.viz_mode.get() == '3D' and z_col and z_col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[z_col]):
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(temp_dates.astype('int64'), self.df[value_col], self.df[z_col], color='purple', lw=1)
            ax.set_xlabel(date_col)
            ax.set_ylabel(value_col)
            ax.set_zlabel(z_col)
            fig.autofmt_xdate()
            ax.set_title(f'3D Time Series: {value_col} vs {z_col} over {date_col}', fontsize=14)
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(self.df[[value_col]])
            anomalies = self.df[(scaled_values > 1.2) | (scaled_values < -0.2)]
            ax.scatter(temp_dates[anomalies.index].astype('int64'), anomalies[value_col], anomalies[z_col], 
                      color='red', label='Anomalies', zorder=5)
            ax.legend()
            ax.grid(True)
            analysis = (f"3D Time Series Explanation:\n"
                       "This 3D time series plots {value_col} and {z_col} over {date_col}. The line shows the trend, "
                       f"with red points marking anomalies (values >1.2 or <-0.2 after scaling). Date range: "
                       f"{temp_dates.min().date()} to {temp_dates.max().date()}. Rotate to explore {z_col}'s impact. "
                       f"Average {value_col}: {self.df[value_col].mean():.2f}, with {len(anomalies)} anomalies.")
        else:
            ax = fig.add_subplot(111)
            sns.lineplot(x=temp_dates, y=self.df[value_col], ax=ax, color='purple', errorbar=None)
            ax.set_title(f'Time Series: {value_col} over {date_col}', fontsize=14, pad=15)
            ax.set_xlabel(date_col)
            ax.set_ylabel(value_col)
            fig.autofmt_xdate()
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(self.df[[value_col]])
            anomalies = self.df[(scaled_values > 1.2) | (scaled_values < -0.2)]
            ax.scatter(temp_dates[anomalies.index], anomalies[value_col], 
                      color='red', label='Anomalies', zorder=5)
            ax.legend()
            ax.grid(True)
            analysis = (f"Time Series Explanation:\n"
                       "This time series plots {value_col} over {date_col}. The line tracks changes over time, with "
                       f"red points marking anomalies (values >1.2 or <-0.2 after scaling). Date range: "
                       f"{temp_dates.min().date()} to {temp_dates.max().date()}. Average {value_col}: "
                       f"{self.df[value_col].mean():.2f}, with {len(anomalies)} anomalies. Look for trends or "
                       "seasonal patterns.")

        self._add_figure(fig, analysis)
        self.status.set(f"{self.viz_mode.get()} Time Series generated successfully")

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
        self.status.set("Insights generated successfully")

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
            
        temp_files = []
        for idx, fig in enumerate(self.current_figures):
            path = f"temp_fig_{idx}.png"
            fig.figure.savefig(path)
            temp_files.append(path)
            pdf.add_page()
            pdf.image(path, x=10, y=10, w=180)
            pdf.ln(100)
            
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf")
        if save_path:
            pdf.output(save_path)
            self.status.set(f"Report exported to {save_path}")
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass

def main():
    print("Initializing GUI...")
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
                messagebox.showerror("Preprocessing Error", f"Failed to preprocess data: {str(e)}")
                print(f"Preprocessing error: {str(e)}")
                root.destroy()

        Button(preprocess_window, text="Apply Preprocessing", command=apply_preprocessing, 
              bg='#4CAF50', fg='white').pack(pady=20)

    except Exception as e:
        messagebox.showerror("Loading Error", f"Failed to load dataset: {str(e)}")
        print(f"Error: {str(e)}")
        root.destroy()

if __name__ == "__main__":
    main()
