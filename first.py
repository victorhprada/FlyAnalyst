import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('flights.csv')

print(f'First 5 rows: {df.head()}')
print(f'Last 5 rows: {df.tail()}')
print(f'Shape: {df.shape}')
print(f'Info: {df.info()}')
print(f'Describe: {df.describe()}')
print(f'Columns: {df.columns}')
print(f'Dtypes: {df.dtypes}')
print(f'Isnull: {df.isnull().sum()}')
print(f'Isnull sum: {df.isnull().sum().sum()}')
print(f'Isnull sum percentage: {df.isnull().sum().sum() / df.size * 100}')
print(f'Isnull sum percentage: {df.isnull().sum().sum() / df.size * 100}')

print(f'Unique values: {df.nunique()}')
print(f'Unique values percentage: {df.nunique() / df.size * 100}')
print(f'Unique values percentage: {df.nunique() / df.size * 100}')

print(f'Describe object: {df.describe(include='O')}')

average_delay = df.groupby('airline')['delay'].mean()
print(f'Average delay: {average_delay}')

average_delay = df.groupby('airline')['delay'].mean().reset_index()
sns.barplot(x='airline', y='delay', data=average_delay)
plt.title('Average delay by airline')
plt.xlabel('Airline')
plt.ylabel('Average delay')
plt.show()

sns.countplot(df, x='airline')
plt.title('Count of flights by airline')
plt.xlabel('Airline')
plt.ylabel('Count')
plt.show()

average_delay = df.groupby('schengen')['delay'].mean().reset_index()
sns.barplot(x='schengen', y='delay', data=average_delay)
plt.title('Average delay by schengen')
plt.xlabel('Schengen')
plt.ylabel('Average delay')
plt.show()

average_delay = df.groupby('is_holiday')['delay'].mean().reset_index()
sns.barplot(x='is_holiday', y='delay', data=average_delay)
plt.title('Average delay by is_holiday')
plt.xlabel('Is holiday')
plt.ylabel('Average delay')
plt.show()

order = df['aircraft_type'].value_counts().index
sns.countplot(df, x='aircraft_type', order=order)
plt.title('Count of flights by aircraft type')
plt.xticks(rotation=70)
plt.xlabel('Aircraft type')
plt.ylabel('Count')
plt.show()

# Bin calculate
def bin_calculate(df, column):
    Q75, Q25 = np.percentile(df[column], [75, 25])
    IQR = Q75 - Q25
    lower_bound = Q25 - 1.5 * IQR
    upper_bound = Q75 + 1.5 * IQR
    larg_bin = 2 * IQR * np.power(len(df[column]), -1/3)
    return lower_bound, upper_bound, larg_bin

lower_bound, upper_bound, larg_bin = bin_calculate(df, 'arrival_time')
print(f'Lower bound: {lower_bound}')
print(f'Upper bound: {upper_bound}')
print(f'Larg bin: {larg_bin}')

# Arrival time distribution
sns.histplot(df, x='arrival_time', binwidth=larg_bin, kde=True)
plt.title('Arrival time distribution')
plt.xlabel('Arrival time')
plt.ylabel('Count')
plt.show()

# Departure time distribution
sns.histplot(df, x='departure_time', binwidth=larg_bin, kde=True)
plt.title('Departure time distribution')
plt.xlabel('Departure time')
plt.ylabel('Count')
plt.show()

mean_delay = df['delay'].mean()
median_delay = df['delay'].median()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(df, y='delay', ax=axes[0])
axes[0].set_title('Delay distribution Box Plot')

axes[0].axhline(y=mean_delay, color='red', linestyle='--', label='Mean')
axes[0].legend()



