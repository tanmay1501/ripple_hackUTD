import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_column_summary(column, dtype):
    """
    Get JSON-friendly summary statistics based on data type
    Returns: JSON formatted dictionary with counts of null, missing, valid values
    """
    summary = {}
    # Check if column has date or time in its name
    if dtype in ['int', 'float', 'numeric','datetime64[ns]','int64','float64']:
        summary = {
            "name": column.name,
            "type": "numeric",
            "data_quality": {
                "total_count": int(len(column)),
                "null_count": int(column.isnull().sum()),
                "valid_count": int(column.notnull().sum())
            },
            "statistics": {
                "mean": float(column.mean()),
                "std": float(column.std()),
                "min": float(column.min()),
                "max": float(column.max())
            }
        }
    

    elif dtype in [ 'category', 'categorical','bool','str']:
        summary = {
            "name": column.name,
            "type": "categorical",
            "data_quality": {
                "total_count": int(len(column)),
                "null_count": int(column.isnull().sum()),
                "valid_count": int(column.notnull().sum())
            },
            "statistics": {
                "unique_count": int(column.nunique()),
                "categories": {
                    str(k): int(v) for k, v in column.value_counts().items()
                },
                "proportions": {
                    str(k): float(v) for k, v in column.value_counts(normalize=True).items()
                }
            }
        }
    
    return summary

def plot_column_distribution(column, dtype):
    """Create visualization based on data type"""
    plt.figure(figsize=(10, 6))
    print(column.name)
    if dtype in ['int', 'float', 'numeric','datetime64[ns]','int64','float64']:
        # Histogram with KDE for numeric data
        sns.histplot(data=column, kde=True)
        plt.title(f'Distribution of {column.name}')
        plt.xlabel(column.name)
        plt.ylabel('Count')
    
    elif dtype in ['object', 'category', 'categorical']:
        # Bar plot for categorical data
        sns.countplot(data=column)
        plt.title(f'Distribution of {column.name}')
        plt.xticks(rotation=45)
        plt.xlabel(column.name)
        plt.ylabel('Count')
    
    plt.tight_layout()
    #convert file to jpg and save it to image folder

    plt.savefig(f'column_distribution_{column.name}.png')
    plt.close()
 
    return f'column_distribution_{column.name}.png'

# Usage example:
"""
# Get summary
summary = get_column_summary(df['price'], 'numeric')
print(summary)

# Get plot
plot = plot_column_distribution(df['price'], 'numeric')
plot.show()
"""