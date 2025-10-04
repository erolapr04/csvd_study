import os, sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import mannwhitneyu, levene, shapiro
from scipy.stats import ttest_ind, f_oneway, kruskal, chi2_contingency, chisquare, linregress
from scipy.stats import sem, t
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import f
import csv
import ast
from typing import List, Any, Union, Iterable, Tuple, Dict

#read data
path = '/Users/e/Desktop/neuro/code/raw files/raw data_excluded.xlsx'
df = pd.read_excel(path)

# columns = df.columns.tolist()
# id = ['ID', 'ID_2', 'Diameter_No']
# groups = ['ICAS', 'CSVD_score', 'SVD_type']
# neumerical = ['age', 'Education_year', 'BA_diameter', 'LICA_diameter', 'RICA_diameter',
#               'Height', 'Weight', 'SBP', 'DBP', 'BMI', 'Waist', 'Hip', 'Pulse', 
#               'Daily_calories ', 'pace_m_per_sec', 'grip', 'chair_sit_up', 'IPAQ',
#               'Cholesterol', 'TG', 'LDL', 'HbA1C', 'MMSE', 
#               'cvlt_30Scorrect', 'cvlt_10Mcorrect', 'Language_BNT_total', 'TY_CFT_copy', 'TY_CFT_delayed',
#               'verbal_fluency_animal_cor', 'digital_forward', 'digital_backward', 'GDS', 'Clock', 'Digit_symbol_120sec', 
#               'GMV', 'WMV', 'CSFV', 'TIV', 'gBA', 'gBAG', 'PVWMH', 'DPWMH', 'TotalWMH']
# categorical = ['CSVD_score', 'SVD_type', 'ICAD_vessels', 'ICAS','sex', 'Alcohol', 'Smoke', 'HTN', 'DM', Dyslipidemia'. 
#                'WMH_score','lacune', 'CMB','strictly_lobar']

class Demography:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def count_missing(self, column_or_list: Union[str, List[Any]]):
        # Accept either a column name or a raw list
        if isinstance(column_or_list, str):
            column_data = list(self.data[column_or_list])
        else:
            column_data = list(column_or_list)
        n = len(column_data)
        clean = [i for i in column_data if i != '"""N/A"""' and pd.notna(i)]
        return clean, len(clean), n - len(clean)

    def mean_confidence_interval(self, column_or_list: Union[str, List[float]], confidence=0.95):
        data, total, missing = self.count_missing(column_or_list)
        mean_val = np.mean(data)
        std_err = sem(data)
        h = std_err * t.ppf((1 + confidence) / 2., total - 1)
        return float(mean_val), float(mean_val - h), float(mean_val + h)

    def median_IQR(self, column_or_list: Union[str, List[float]]):
        data, total, missing = self.count_missing(column_or_list)
        if total < 2:
            return float('nan'), float('nan'), float('nan')
        median_val = float(np.median(data))
        q1 = float(np.percentile(data, 25))
        q3 = float(np.percentile(data, 75))
        return median_val, q1, q3

    def count_percentage(self, column_or_list: Union[str, List[Any]]):
        data, total, missing = self.count_missing(column_or_list)
        counts = pd.Series(data).value_counts(dropna=False).sort_index()
        pct = counts / total * 100
        return {
            'Category': counts.index.tolist(),
            'Count': counts.astype(int).tolist(),
            'Percentage': pct.round(1).tolist()
        }

#trying = Demography(df)
#xxx = trying.count_missing('sex')
#xxx = trying.mean_confidence_interval('Education_year')
#xxx = trying.median_IQR('verbal_fluency_animal_cor')
#xxx = trying.count_percentage('HTN')
#print(xxx)

class StatsMethods:
    def __init__(self, data):
        self.data = data

    #data = [(1,1,1), (1,1,1), (1,1,1)]
    def test_normality(self, data):
        for i in data:
            if len(i) < 3:
                return False
            try:
                stat, p = shapiro(i)
            except Exception:
                return False
            if p < 0.05:
                return False
        return True

    def test_homogeneity(self, data):
        stat, p = levene(*data)
        if p >= 0.05 :
            return False
        else:
            return True

    def cohen_d(self, data1, data2):
        n1, n2 = len(data1), len(data2)
        s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        d = (np.mean(data1) - np.mean(data2)) / pooled_std
        return d

    def t_u_test(self, data):
        normality = self.test_normality(data)
        homogeneity = self.test_homogeneity(data)
        if normality and homogeneity:
            stat, p = ttest_ind(data[0], data[1])
            return stat, p, 't-test'
        else:
            stat, p = mannwhitneyu(data[0], data[1])
            return stat, p, 'Mann-Whitney U'

    def multi_group_test(self, data):
        normality = self.test_normality(data)
        homogeneity = self.test_homogeneity(data)
        if normality and homogeneity:
            # Use one-way ANOVA
            stat, p = f_oneway(*data)
            return stat, p, 'ANOVA'
        else:
            # Use Kruskal-Wallis test
            stat, p = kruskal(*data)
            return stat, p, 'Kruskal-Wallis'

    def chi2_test_2groups(self, data1: Iterable, data2: Iterable) -> Tuple[float, float]:
        table = pd.crosstab(list(data1), list(data2))
        stat, p, dof, expected = chi2_contingency(table)
        return stat, p

    def chi2_test_multigroup(self, data: Iterable) -> Tuple[float, float]:
        counts = pd.Series(data).value_counts()
        observed = counts.values
        expected = [observed.sum() / len(observed)] * len(observed)
        stat, p = chisquare(f_obs=observed, f_exp=expected)
        return stat, p

    def linear_regression(self, x: Iterable[float], y: Iterable[float]) -> Tuple[float, float, float]:
        result = linregress(x, y)
        return result.slope, result.rvalue, result.pvalue
    
    def multivariate_linear_regression(self, X: pd.DataFrame, y: pd.Series, standardize: bool = False) -> Tuple[str, float, float]:
        data = pd.concat([X, y], axis=1).dropna()
        X_clean = data[X.columns].astype(float)
        y_clean = data[y.name].astype(float)

        if standardize:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_clean)
            y_scaled = scaler_y.fit_transform(y_clean.values.reshape(-1, 1)).ravel()
        else:
            X_scaled = X_clean.values
            y_scaled = y_clean.values

        n = X_scaled.shape[0]
        p = X_scaled.shape[1]

        model = LinearRegression()
        model.fit(X_scaled, y_scaled)
        coefficients = model.coef_
        intercept = model.intercept_
        r_squared = model.score(X_scaled, y_scaled)

        y_pred = model.predict(X_scaled)
        residuals = y_scaled - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_total = np.sum((y_scaled - np.mean(y_scaled)) ** 2)

        # Mean squared error of residuals
        ms_res = ss_res / (n - p - 1)
        ms_reg = (ss_total - ss_res) / p

        # Overall model F-test p-value
        f_stat = ms_reg / ms_res
        overall_p_value = 1 - f.cdf(f_stat, p, n - p - 1)

        # Compute variance-covariance matrix of coefficients
        X_with_intercept = np.column_stack([np.ones(n), X_scaled])
        XtX_inv = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept))
        se = np.sqrt(np.diagonal(XtX_inv) * ms_res)  # standard errors including intercept

        # t-statistics for coefficients (exclude intercept se for coeff p-values)
        t_stats = coefficients / se[1:]  # skip intercept std error
        p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=n - p - 1))  # two-sided p-values
        demographic_vars = ['age', 'sex', 'Education_year']
        terms = [f"{coeff:.4f}*{name} (p={pval:.4g})" for coeff, name, pval in sorted(
        zip(coefficients, X.columns, p_values),
        key=lambda item: item[1] in demographic_vars
        )]   
        formula = f"{y.name} = {intercept:.4f} + " + " + ".join(terms)
        target_p = -np.log10(next(pval for name, pval in zip(X.columns, p_values) if name not in demographic_vars))

        return formula, r_squared, overall_p_value, coefficients, p_values, target_p

    @staticmethod
    def get_regression_results_for_plot(X: pd.DataFrame, y: pd.Series, target_variable: str,alpha: float = 0.05) -> Dict:

        # 1. 清理數據：移除任何包含缺失值的行
        data = pd.concat([X, y], axis=1).dropna()
        if data.empty:
            return None # 如果沒有數據，則返回 None
            
        X_clean = data[X.columns].astype(float)
        y_clean = data[y.name].astype(float)

        # 確保數據足夠進行迴歸
        n = X_clean.shape[0]
        p = X_clean.shape[1]
        if n <= p:
            print(f"警告：'{y.name}' 的樣本數 ({n}) 不足以進行迴歸分析 (需要 > {p})。")
            return None

        model = LinearRegression()
        model.fit(X_clean, y_clean)
        coefficients = model.coef_

        # 3. 計算標準誤 (Standard Error) 和 P-value
        # 這部分邏輯與您原始的程式碼相同
        residuals = y_clean - model.predict(X_clean)
        ss_res = np.sum(residuals ** 2)
        ms_res = ss_res / (n - p - 1)

        X_with_intercept = np.column_stack([np.ones(n), X_clean.values])
        try:
            XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        except np.linalg.LinAlgError:
            print(f"警告：'{y.name}' 的矩陣計算出錯，可能是共線性問題。")
            return None
            
        se = np.sqrt(np.diagonal(XtX_inv) * ms_res)
        se_coeffs = se[1:] # 移除截距項的標準誤

        t_stats = coefficients / se_coeffs
        p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=n - p - 1))

        # Count CI
        df = n - p - 1 
        t_critical = t.ppf(1 - alpha / 2, df=df)
        margin_of_error = t_critical * se_coeffs
        ci_lower_all = coefficients - margin_of_error
        ci_upper_all = coefficients + margin_of_error
        
        # 5. 提取目標變項的結果
        try:
            target_idx = X.columns.get_loc(target_variable)
        except KeyError:
            raise ValueError(f"錯誤：目標變項 '{target_variable}' 不在 X 的欄位中。")

        beta = coefficients[target_idx]
        p_value = p_values[target_idx]
        ci_lower = ci_lower_all[target_idx]
        ci_upper = ci_upper_all[target_idx]

        # 6. 判斷是否顯著並整理輸出
        is_significant = "Significant" if p_value < alpha else "Not Significant"

        plot_result = {
            'test_name': y.name,
            'beta': beta,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'is_significant': is_significant
        }
        
        return plot_result

class Export_Table1:
    def __init__(self, df: pd.DataFrame, csvd_merge=False, categorical_merge=False, normalize=False):
        self.df = df
        if csvd_merge == True:
            remap_dict = {
                0: "0/1",
                1: "0/1",
                2: "2/3",
                3: "2/3"
            }
            self.df['CSVD_score'] = self.df['CSVD_score'].replace(remap_dict)
        if categorical_merge == True:
            for col in ['ICAD_vessels', 'ICAS', 'sex', 'Alcohol', 'Smoke', 'HTN', 'DM', 'Dyslipidemia']:
                self.df.loc[self.df[col] >= 1, col] = 1
        if normalize == True:
            for column in ['BA_diameter', 'LICA_diameter', 'RICA_diameter']:
                numerator = pd.to_numeric(self.df[column], errors='coerce')
                denominator = pd.to_numeric(self.df['TIV'], errors='coerce')
                valid = (denominator != 0) & denominator.notna()
                self.df[column] = '"""N/A"""'
                self.df.loc[valid, column] = numerator[valid] / denominator[valid]
        self.dem = Demography(self.df)
        self.stats = StatsMethods(df)
        self.numerical = [
            'age','Education_year','BA_diameter', 'LICA_diameter', 'RICA_diameter',
            'Height','Weight','SBP','DBP','BMI','Waist','Hip',
            'Pulse','Daily_calories ','gait','grip','sit_stand','IPAQ',
            'Cholesterol','TG','LDL','HbA1C','MMSE','cvlt_30Scorrect','cvlt_10Mcorrect',
            'Language_BNT_total','TY_CFT_copy','TY_CFT_delayed','verbal_fluency_animal_cor',
            'digital_forward','digital_backward','GDS','Clock','Digit_symbol_120sec',
            'GMV','WMV','CSFV','TIV','gBA','gBAG','PVWMH','DPWMH','TotalWMH', 
        ]
        self.categorical = ['CSVD_score','SVD_type', 'ICAD_vessels', 'ICAS', 'sex','Alcohol','Smoke','HTN', 'DM','Dyslipidemia',
                            'WMH_score','lacune', 'CMB','strictly_lobar']
        self.group_cols = ['CSVD_score']
        self.groups = {col: sorted(self.df[col].dropna().unique()) for col in self.group_cols}

    def _build_header(self) -> List[str]:
        hdr = ['Feature','Total','Missing','All','Parametric']
        for col in self.group_cols:
            for lvl in self.groups[col]:
                hdr.append(f"{col}_{(lvl)}")
        return hdr

    def _summarize_numeric(self, values: List[float]) -> tuple:
        clean, count, _ = self.dem.count_missing(values)
        is_norm = self.stats.test_normality([clean])
        if is_norm:
            mean, lo, hi = self.dem.mean_confidence_interval(clean)
            return f"{mean:.2f} [{lo:.2f}–{hi:.2f}] (n={count})", True
        med, q1, q3 = self.dem.median_IQR(clean)
        return f"{med:.2f} [{q1:.2f}–{q3:.2f}] (n={count})", False

    def _summarize_categorical(self, column_or_list: Union[str, List[Any]]) -> str:
        summary = self.dem.count_percentage(column_or_list)
        cats = summary['Category']
        counts = summary['Count']
        pcts = summary['Percentage']
        return "; ".join(f"{cats[i]}:{counts[i]}({pcts[i]}%)" for i in range(len(cats)))

    def summarize_all(self, feat: str) -> tuple:
        clean_all, valid_all, miss_all = self.dem.count_missing(feat)
        if feat in self.numerical:
            all_s, is_param = self._summarize_numeric(clean_all)
        else:
            if feat == 'CSVD_score' or 'SVD_type':
                all_s = self._summarize_categorical(feat)
            else:
                all_s = self._summarize_categorical(clean_all)
            is_param = False
        return valid_all, miss_all, all_s, is_param

    def summarize_groups(self, feat: str) -> List[str]:
        _, _, _, _ = self.summarize_all(feat)
        cells = []
        for col in self.group_cols:
            for lvl in self.groups[col]:
                subset = list(self.df[self.df[col] == lvl][feat])
                if feat in self.numerical:
                    summary, _ = self._summarize_numeric(subset)
                else:
                    summary = self._summarize_categorical(subset)
                cells.append(summary)
        return cells

    def table1(self) -> List[List[Any]]:
        header = self._build_header()
        rows = []
        for feat in self.numerical + self.categorical:
            total, miss, all_summary, is_param = self.summarize_all(feat)
            group_cells = self.summarize_groups(feat)
            rows.append([feat, total, miss, all_summary, is_param] + group_cells)
        return [header] + rows

    def export_table1(self, path: str) -> None:
        table = self.table1()
        df_out = pd.DataFrame(table[1:], columns=table[0])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df_out.to_csv(path, index=False)

class Export_StatResults:
    def __init__(self, df: pd.DataFrame, group_cols: List[str], csvd_merge=False, categorical_merge=False, normalize=False):
        self.df = df
        if csvd_merge == True:
            remap_dict = {
                0: "0/1",
                1: "0/1",
                2: "2/3",
                3: "2/3"
            }
            self.df['CSVD_score'] = self.df['CSVD_score'].replace(remap_dict)
        if categorical_merge == True:
            for col in ['ICAD_vessels', 'ICAS', 'sex','Alcohol','Smoke','HTN', 'DM','Dyslipidemia']:
                self.df.loc[self.df[col] >= 1, col] = 1
        if normalize == True:
            for column in ['BA_diameter', 'LICA_diameter', 'RICA_diameter']:
                numerator = pd.to_numeric(self.df[column], errors='coerce')
                denominator = pd.to_numeric(self.df['TIV'], errors='coerce')
                valid = (denominator != 0) & denominator.notna()
                self.df[column] = '"""N/A"""'
                self.df.loc[valid, column] = numerator[valid] / denominator[valid]
        self.dem = Demography(df)
        self.stats = StatsMethods(df)
        self.group_cols = group_cols
        self.numerical = [
            'age','Education_year','BA_diameter', 'LICA_diameter', 'RICA_diameter',
            'Height','Weight','SBP','DBP','BMI','Waist','Hip',
            'Pulse','Daily_calories ','gait','grip','sit_stand','IPAQ',
            'Cholesterol','TG','LDL','HbA1C','MMSE','cvlt_30Scorrect','cvlt_10Mcorrect',
            'Language_BNT_total','TY_CFT_copy','TY_CFT_delayed','verbal_fluency_animal_cor',
            'digital_forward','digital_backward','GDS','Clock','Digit_symbol_120sec',
            'GMV','WMV','CSFV','TIV','gBA','gBAG','PVWMH','DPWMH','TotalWMH', 
        ]
        self.categorical = ['CSVD_score','SVD_type', 'ICAD_vessels', 'ICAS', 'sex','Alcohol','Smoke','HTN', 'DM','Dyslipidemia',
                            'WMH_score','lacune', 'CMB','strictly_lobar']

    def _prepare_groups(self, group_col: str) -> List[Any]:
        return sorted(self.df[group_col].dropna().unique())

    def _get_group_data(self, feat: str, group_col: str) -> List[List[Any]]:
        groups = self._prepare_groups(group_col)
        data = []
        for lvl in groups:
            subset = self.df[self.df[group_col] == lvl][feat]
            clean, _, _ = self.dem.count_missing(subset)
            data.append(clean)
        return data

    def _test_numeric(self, feat: str, group_col: str) -> tuple:
        data = self._get_group_data(feat, group_col)
        num_groups = len(data)
        if num_groups == 2:
            stat, p, test_name = self.stats.t_u_test(data)
        else:  
            stat, p, test_name = self.stats.multi_group_test(data)
            
        return stat, p, test_name

    def _test_categorical(self, feat: str, group_col: str) -> tuple:
        sub_df = self.df[[feat, group_col]].dropna()
        stat, p = self.stats.chi2_test_2groups(sub_df[feat], sub_df[group_col])
        return stat, p

    def run_experiment(self, group_col: str) -> pd.DataFrame:
        results = []
        for feat in self.numerical + self.categorical:
            if feat in self.numerical:
                stat, p, test_name = self._test_numeric(feat, group_col)
                results.append({
                    'Feature': feat,
                    'Test': test_name,
                    'Statistic': stat,
                    'p-value': p
                })
            else:
                stat, p = self._test_categorical(feat, group_col)
                results.append({
                    'Feature': feat,
                    'Test': 'Chi-square',
                    'Statistic': stat,
                    'p-value': p
                })
        return pd.DataFrame(results)

    def run_all_experiments(self) -> dict:
        return {col: self.run_experiment(col) for col in self.group_cols}

    def export_results(self, path: str, group_col: str) -> None:
        df_results = self.run_experiment(group_col)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df_results.to_csv(path, index=False)

class Export_DiametervsCategorical:
    def __init__(self, df: pd.DataFrame, group_cols: List[str], csvd_merge=False, categorical_merge=False, normalize=False):
        self.df = df
        if csvd_merge == True:
            self.df['CSVD_score'] = self.df['CSVD_score'].replace(['CSVD score 2','CSVD score 3'], 'CSVD score 1')
        if categorical_merge == True:
            for col in ['ICAD_vessels', 'ICAS', 'sex','Alcohol','Smoke','HTN', 'DM','Dyslipidemia','lacune', 'CMB','strictly_lobar']:
                self.df.loc[self.df[col] >= 1, col] = 1
        if normalize == True:
            for column in ['BA_diameter', 'LICA_diameter', 'RICA_diameter']:
                numerator = pd.to_numeric(self.df[column], errors='coerce')
                denominator = pd.to_numeric(self.df['TIV'], errors='coerce')
                valid = (denominator != 0) & denominator.notna()
                self.df[column] = '"""N/A"""'
                self.df.loc[valid, column] = numerator[valid] / denominator[valid]
        self.stats = StatsMethods(df)
        self.dem = Demography(df)
        self.group_cols = group_cols
        self.diameter = ['BA_diameter', 'LICA_diameter', 'RICA_diameter']
        self.categorical = ['ICAS', 'sex', 'Alcohol' ,'Smoke','HTN', 'DM','Dyslipidemia','lacune', 'CMB','strictly_lobar']

    def _get_two_groups(self, feat: str, cat_col: str, df_subset: pd.DataFrame) -> Tuple[List[float], List[float]]:
        clean0, _, _ = Demography(df_subset[df_subset[cat_col] == 0]).count_missing(feat)
        clean1, _, _ = Demography(df_subset[df_subset[cat_col] == 1]).count_missing(feat)
        group0 = [float(i) for i in clean0]
        group1 = [float(i) for i in clean1]
        return group0, group1

    def _summarize_diameter(self, data: list):
        is_norm = self.stats.test_normality([data])
        if is_norm:
            mean, lo, hi = self.dem.mean_confidence_interval(data)
            return f"{mean:.2f} [{lo:.2f}–{hi:.2f}]", True
        med, q1, q3 = self.dem.median_IQR(data)
        return f"{med:.2f} [{q1:.2f}–{q3:.2f}]", False

    def _test_numeric(self, feat: str, cat_col: str, df_subset: pd.DataFrame) -> Tuple[float, float, str]:
        group0_data = df_subset[df_subset[cat_col] == 0][feat].tolist()
        group1_data = df_subset[df_subset[cat_col] == 1][feat].tolist()
        clean0, _, _ = self.dem.count_missing(group0_data)
        clean1, _, _ = self.dem.count_missing(group1_data)
        summary0, _ = self._summarize_diameter(clean0)
        summary1, _ = self._summarize_diameter(clean1)
        n0, n1 = len(clean0), len(clean1)
        if n0 == 0 or n1 == 0:
            return float('nan'), float('nan'), 'Insufficient data', n0, n1, summary0, summary1
        stat, p, test_name = self.stats.t_u_test((clean0, clean1))
        return stat, p, test_name, n0, n1, summary0, summary1

    def run_analysis(self) -> pd.DataFrame:
        results = []
        for cat_col in self.categorical:
            for diam in self.diameter:
                stat, p, test, n0, n1, summary0, summary1 = self._test_numeric(diam, cat_col, self.df)
                results.append({
                    'group_val': 'All',
                    'cat_col': cat_col,
                    'diameter': diam,
                    'group0': summary0,
                    'group1': summary1,
                    'n_group0': n0,
                    'n_group1': n1,
                    'test': test,
                    'statistic': stat,
                    'p_value': p,

                })
        # For each subgroup
        for group_col in self.group_cols:
            groups = sorted(self.df[group_col].dropna().unique())
            for group_val in groups:
                df_sub = self.df[self.df[group_col] == group_val]
                for cat_col in self.categorical:
                    for diam in self.diameter:
                        stat, p, test, n0, n1, summary0, summary1 = self._test_numeric(diam, cat_col, df_sub)
                        results.append({
                            'group_val': group_val,
                            'cat_col': cat_col,
                            'diameter': diam,
                            'group0': summary0,
                            'group1': summary1,
                            'n_group0': n0,
                            'n_group1': n1,                            
                            'test': test,
                            'statistic': stat,
                            'p_value': p,

                        })
        return pd.DataFrame(results)

    def export_results(self, path: str, df_results: pd.DataFrame) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df_results.to_csv(path, index=False)

class Export_Appendix:
    def __init__(self, df: pd.DataFrame, csvd_merge=False, categorical_merge=False, normalize=False):
        self.df = df
        if csvd_merge == True:
            self.df['CSVD_score'] = self.df['CSVD_score'].replace('CSVD score 3', 'CSVD score 2')
        if categorical_merge == True:
            for col in ['ICAD_vessels', 'ICASS', 'sex','Alcohol','Smoke','HTN', 'DM','Dyslipidemia']:
                self.df.loc[self.df[col] >= 1, col] = 1
        if normalize == True:
            for column in ['BA_diameter', 'LICA_diameter', 'RICA_diameter']:
                numerator = pd.to_numeric(self.df[column], errors='coerce')
                denominator = pd.to_numeric(self.df['TIV'], errors='coerce')
                valid = (denominator != 0) & denominator.notna()
                self.df[column] = '"""N/A"""'
                self.df.loc[valid, column] = numerator[valid] / denominator[valid]
        self.dem = Demography(df)
        self.stats = StatsMethods(df)
        self.diameter = ['BA_diameter', 'LICA_diameter', 'RICA_diameter']
        self.marker = ['WMH_score', 'lacune', 'CMB', 'strictly_lobar']
        
    def _get_data(self, marker: str, diameter_col: str):
        data0 = self.df[self.df[marker] == 0][diameter_col]
        clean0, n0, missing0 = self.dem.count_missing(data0)

        data1 = self.df[self.df[marker] == 1][diameter_col]
        clean1, n1, missing1 = self.dem.count_missing(data1)
        
        clean = [clean0, clean1]
        counts = [n0, n1]
        missings = missing0+missing1
        
        return clean, counts, missings

    def _run_numerical(self, marker: str, diameter_col: str):
        data, n, _ = self._get_data(marker, diameter_col)
        stat, p, test = self.stats.t_u_test(data)
        med1, _, _ = self.dem.median_IQR(data[0])
        med2, _, _ = self.dem.median_IQR(data[1])
        # print(med1, med2, n[0], n[1], stat, p, test)
        return med1, med2, n[0], n[1], stat, p, test

    def run_diameter_in_marker(self):
        results=[]
        for vessel in self.diameter:
            for marker in self.marker:
                med1, med2, n0, n1, stat, p, test = self._run_numerical(marker, vessel)
                results.append({
                    'Group': 'All',
                    'marker': marker,
                    'diameter': vessel,
                    'median_0': med1,
                    'median_1': med2,
                    'n_0': n0,
                    'n_1': n1,
                    'statistic': stat,
                    'p_value': p,
                    'test': test
                })

        groups = sorted(self.df['SVD_type'].dropna().unique())
        for group_val in groups:
            df_sub = self.df[self.df['SVD_type'] == group_val]
            for vessel in self.diameter:
                for marker in self.marker:
                    med1, med2, n0, n1, stat, p, test = self._run_numerical(marker, vessel)
                    results.append({
                        'Group': group_val,
                        'marker': marker,
                        'diameter': vessel,
                        'median_0': med1,
                        'median_1': med2,
                        'n_0': n0,
                        'n_1': n1,
                        'statistic': stat,
                        'p_value': p,
                        'test': test
                    })

        return pd.DataFrame(results)

    def export(self, path: str, df_results: pd.DataFrame) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df_results.to_csv(path, index=False)

if __name__ == "__main__":
    # exporter = Export_Table1(df, csvd_merge=True, categorical_merge=True, normalize=True)
    # exporter.export_table1('/Users/e/Desktop/neuro/results/0726/table1.csv')

    # group_cols = ['ICAS']
    # exporter = Export_StatResults(df, group_cols, csvd_merge=True, categorical_merge=True, normalize=False)
    # results = exporter.run_all_experiments()
    # exporter.export_results('/Users/e/Desktop/neuro/results/0726/table1stats.csv', 'ICAS')
    
    exporter = Export_Table1(df, csvd_merge=True, categorical_merge=True, normalize=True)
    exporter.export_table1('/Users/e/Desktop/neuro/results/0814/counts_merged.csv')
    
    # group_cols = ['CSVD_score']
    # exporter = Export_StatResults(df, group_cols, csvd_merge=True, categorical_merge=True, normalize=True)
    # results = exporter.run_all_experiments()
    # exporter.export_results('/Users/e/Desktop/neuro/results/0814/stats_merged', 'CSVD_score')


    # exporter = Export_DiametervsCategorical(df, group_cols, csvd_merge=True, categorical_merge=True, normalize=True)
    # results_df = exporter.run_analysis()
    # exporter.export_results('/Users/chenweida/opt/anaconda3/Neuro/Results/diameter_categorical_results.csv', results_df)

    # exporter = Export_Appendix(df, csvd_merge=False, categorical_merge=False, normalize=True)
    # results = exporter.run_diameter_in_marker()
    # exporter.export('/Users/chenweida/opt/anaconda3/Neuro/Results/marker.csv', results)
    
    # testing= Export(df)
    # print(testing.summarize_all('CSVD_score')
