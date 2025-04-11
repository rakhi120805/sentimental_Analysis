import pandas as pd

# Load data
merged_data = pd.read_csv("tsla_merged_data.csv")
pca_results = pd.read_csv("tsla_pca_results.csv")
# Load loadings, set index to first column (features), and convert to numeric
loadings = pd.read_csv("tsla_pca_loadings.csv", index_col=0)
loadings = loadings.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, NaN for failures

# Interpret PCA
num_pcs = len(loadings.columns)
print(f"Number of Principal Components: {num_pcs}")
for pc in loadings.columns:
    top_features = loadings[pc].abs().sort_values(ascending=False).head(3)
    print(f"\n{pc} Top Features:")
    print(top_features)

# Summary
summary = """
Tesla Stock and Sentiment Analysis (March 12 - April 10, 2025):
- Stock data: Daily Open, Close, High, Low, Volume from yFinance.
- Sentiment: NewsAPI data with VADER sentiment (mostly neutral except April 10).
- PCA: Reduced 8 features to {num_pcs} components.
- PC1 likely captures price trends (Close, Open).
- PC2 may reflect trading activity (Volume, Volatility).
- Sentiment impact limited due to sparse news data.
""".format(num_pcs=num_pcs)

# Move replace outside f-string
summary_html = summary.replace('\n', '<br>')

# Generate HTML website
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Tesla Stock and Sentiment Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 80%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 80%; height: auto; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Tesla Stock and Sentiment Analysis</h1>
    <h3>Period: March 12 - April 10, 2025</h3>
    <p>{summary_html}</p>

    <h2>Sample Data</h2>
    <table>
        <tr>
            <th>Date</th>
            <th>Close Price</th>
            <th>Sentiment</th>
            <th>Daily Return</th>
            <th>PC1</th>
        </tr>
"""

# Add sample data (first 5 rows)
for i, row in merged_data.head().iterrows():
    pc1 = pca_results["PC1"].iloc[i] if "PC1" in pca_results.columns else 0
    html_content += f"""
        <tr>
            <td>{row['Date']}</td>
            <td>{row['Close']:.2f}</td>
            <td>{row['Sentiment']:.4f}</td>
            <td>{row['Daily_Return']:.4f}</td>
            <td>{pc1:.2f}</td>
        </tr>
    """

# Add PCA visuals and loadings
html_content += """
    </table>

    <h2>PCA Visualizations</h2>
    <img src="scree_plot.png" alt="Scree Plot">
    <img src="loadings_heatmap.png" alt="Loadings Heatmap">

    <h2>PCA Loadings</h2>
    <table>
        <tr>
            <th>Feature</th>
"""

# Add PC column headers
for pc in loadings.columns:
    html_content += f"<th>{pc}</th>"
html_content += "</tr>"

# Add loadings data
for feature, row in loadings.iterrows():
    html_content += f"<tr><td>{feature}</td>"
    for value in row:
        html_content += f"<td>{value:.3f}</td>"
    html_content += "</tr>"

html_content += """
    </table>
</body>
</html>
"""

# Save HTML file
with open("tesla_analysis.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("\nWebsite generated: tesla_analysis.html")
print(summary)