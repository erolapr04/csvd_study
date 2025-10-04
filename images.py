import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

path = '/Users/e/Desktop/neuro/effector_gait.xlsx'
df = pd.read_excel(path)

df = df.rename(columns={'CSVD_score': 'csvd_score', 'age': 'age', 'sex': 'sex', 'gait': 'gait'})
df['csvd_group_named'] = np.where(df['csvd_score'] >= 2, 'High', 'Low')

formula = "gait ~ C(csvd_group_named, Treatment(reference='Low')) + age + sex"
model = smf.ols(formula, data=df).fit()

age_coef = model.params['age']
sex_coef = model.params['sex']
df['gait_adjusted'] = df['gait'] - (age_coef * df['age']) - (sex_coef * df['sex'])

fig, ax = plt.subplots(figsize=(10, 8))

# 圖層 1: 小提琴圖
sns.violinplot(
    x='csvd_group_named', y='gait_adjusted', data=df,
    order=['High','Low'], palette=['#FFADAD', '#ADDEFF'],
    linewidth=0.8, 
    alpha=0.9,
    inner=None,
    ax=ax
)

# 修改小提琴圖為半邊
for i, violin in enumerate(ax.collections):
    vertices = violin.get_paths()[0].vertices
    center = vertices[:, 0].mean()
    
    if i == 0:  # High group
        vertices[vertices[:, 0] <= center, 0] = center
    else:  # Low group
        vertices[vertices[:, 0] <= center, 0] = center

# 圖層 2: 散點圖 (手動調整位置)
offset = 0.17  # 左移距離

for i, group in enumerate(['High', 'Low']):
    group_data = df[df['csvd_group_named'] == group]
    
    # 添加隨機抖動
    x_positions = np.full(len(group_data), i) - offset
    x_jitter = np.random.normal(0, 0.05, len(group_data))
    x_positions += x_jitter
    
    ax.scatter(x_positions, group_data['gait_adjusted'], 
              color='gray', alpha=0.7, s=30, linewidth=0.5, edgecolor='white')

# 圖層 3: 箱形圖 (手動調整位置)
for i, group in enumerate(['High', 'Low']):
    group_data = df[df['csvd_group_named'] == group]['gait_adjusted']
    
    # 計算箱形圖統計量
    q1, median, q3 = np.percentile(group_data, [25, 50, 75])
    iqr = q3 - q1
    whisker_low = max(group_data.min(), q1 - 1.5 * iqr)
    whisker_high = min(group_data.max(), q3 + 1.5 * iqr)
    
    x_pos = i - offset
    box_width = 0.1
    
    # 畫箱子
    ax.add_patch(plt.Rectangle((x_pos - box_width/2, q1), box_width, iqr, 
                              facecolor='none', edgecolor='black', linewidth=1))
    
    # 畫中位數線
    ax.plot([x_pos - box_width/2, x_pos + box_width/2], [median, median], 
            color='black', linewidth=1.5)
    
    # 畫鬚線
    ax.plot([x_pos, x_pos], [q3, whisker_high], color='black', linewidth=1)
    ax.plot([x_pos, x_pos], [q1, whisker_low], color='black', linewidth=1)
    ax.plot([x_pos - box_width/4, x_pos + box_width/4], [whisker_high, whisker_high], 
            color='black', linewidth=1)
    ax.plot([x_pos - box_width/4, x_pos + box_width/4], [whisker_low, whisker_low], 
            color='black', linewidth=1)

ax.set_title("Distribution of Adjusted Gait Speed by CSVD Group", fontsize=20, pad=20)
ax.set_xlabel("CSVD Group", fontsize=16)
ax.set_ylabel("Gait Speed (Adjusted for Age and Sex)", fontsize=16)

output_directory = '/Users/e/Desktop/neuro/results/plots'
os.makedirs(output_directory, exist_ok=True)
plt.savefig(os.path.join(output_directory, 'adjusted_gait_seaborn_combo_plot.png'), dpi=300, bbox_inches='tight')
plt.show()

print("圖表已產生並儲存完畢。")