# Netflix Content Analysis and User Engagement Insights Using Clustering

An unsupervised machine learning project exploring Netflix's content catalogue through exploratory data analysis, feature engineering, and clustering techniques to identify patterns in content characteristics and derive potential audience-engagement insights.

## Project Overview

Streaming platforms such as Netflix offer a diverse catalogue of movies and television shows. Understanding how content differs by genre, release year, country, duration, and other attributes can help reveal catalogue patterns and inform content strategy.

This project applies **exploratory data analysis (EDA)** and **unsupervised clustering** to discover groups of similar Netflix titles. Multiple clustering algorithms are evaluated to examine which approach produces the most distinct groupings in the dataset.

The project focuses on discovering content patterns and exploring how these patterns could support content categorization, catalogue analysis, and engagement-oriented business hypotheses.

> **Important:** Clustering identifies similarities between content items. Unless actual viewing, watch-time, completion, or engagement labels are included and evaluated, the resulting clusters should be interpreted as content segments—not direct predictions of user engagement.

## Objectives

- Analyze the composition and characteristics of Netflix's content catalogue.
- Explore trends in content types, genres, release years, countries, and duration.
- Prepare relevant features for unsupervised machine learning.
- Apply and compare DBSCAN, K-Means, and Agglomerative Clustering.
- Evaluate clustering quality using the silhouette score where applicable.
- Investigate how content segments could inform catalogue organization and engagement-related analysis.
- Translate analytical findings into potential business applications and future research directions.

## Dataset

The project uses a Netflix titles dataset containing metadata about movies and television shows.

Typical fields in this type of dataset may include:

- `show_id` — content identifier.
- `type` — movie or TV show.
- `title` — title name.
- `director` — director information.
- `cast` — cast information.
- `country` — country associated with the title.
- `date_added` — date the title was added to the catalogue.
- `release_year` — original release year.
- `rating` — content classification.
- `duration` — movie runtime or number of TV show seasons.
- `listed_in` — genres or categories.
- `description` — text description of the title.

The exact columns and preprocessing steps should be verified against the dataset used in the notebook.

The dataset is not included in this repository. If you use a publicly available Netflix titles dataset, provide its original source and review its licence and terms of use before redistribution.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Development environment | Google Colab / Jupyter Notebook |
| Data manipulation | pandas, NumPy |
| Data visualization | Matplotlib, Seaborn |
| Machine learning | scikit-learn |
| Clustering algorithms | DBSCAN, K-Means, Agglomerative Clustering |
| Evaluation | Silhouette score and cluster analysis |

## Methodology

### 1. Data Loading and Preparation

The dataset is loaded into a pandas DataFrame and inspected to understand its structure, feature types, missing values, and potential data-quality issues.

Data preparation may include handling missing values, removing duplicate records where appropriate, converting data types, and preparing relevant features for analysis.

### 2. Exploratory Data Analysis

EDA is used to understand the catalogue's composition and identify patterns worth investigating.

Potential areas of exploration include:

- Distribution of movies versus television shows.
- Content releases across years.
- Most frequently represented genres.
- Country-level content distribution.
- Movie durations and TV show season counts.
- Patterns in content ratings and catalogue additions.

Visualizations help communicate these patterns and identify opportunities for more detailed analysis.

### 3. Feature Engineering

Relevant content attributes are prepared for clustering. Depending on the implemented notebook, this may involve encoding categorical variables, scaling numerical features, and transforming selected metadata into numerical representations.

Feature selection and preprocessing are important because clustering algorithms are sensitive to the representation and scale of the input data.

### 4. Clustering Models

Three unsupervised learning algorithms are compared.

**DBSCAN**

Density-Based Spatial Clustering of Applications with Noise groups densely connected observations and identifies observations in sparse regions as noise. It can discover irregularly shaped clusters without requiring the number of clusters to be specified in advance.

**K-Means**

K-Means partitions observations into a specified number of clusters by iteratively assigning points to centroids and updating those centroids. It provides a useful baseline for segmenting content based on numerical feature representations.

**Agglomerative Clustering**

Agglomerative Clustering builds a hierarchy of clusters by progressively merging observations or groups according to a linkage criterion. It provides another perspective on similarities between content items.

### 5. Model Evaluation

The silhouette score is used to compare cluster separation and cohesion where the metric is mathematically applicable.

A higher silhouette score generally indicates that observations are closer to their own clusters than to neighboring clusters. However, scores should be interpreted alongside the number and size of clusters, DBSCAN noise points, parameter settings, and the practical meaning of the resulting groups.

## Results

### Clustering comparison

The following scores were obtained in the reported experiments:

| Algorithm | Silhouette Score |
|---|---:|
| DBSCAN | **0.7970** |
| K-Means | 0.6781 |
| Agglomerative Clustering | 0.6638 |

DBSCAN achieved the highest reported silhouette score among the three approaches.

### Key findings

- DBSCAN produced the strongest silhouette score in the current experiments.
- K-Means and Agglomerative Clustering provided comparison baselines for examining alternative content groupings.
- The results demonstrate that clustering algorithms can produce different segmentation structures depending on their assumptions, parameters, and input features.
- The practical value of each segmentation depends on cluster size, noise handling, feature selection, and how clearly the groups can be interpreted.

**Interpretation note:** The silhouette score alone does not establish that a clustering model is universally superior. DBSCAN's treatment of noise and the precise observations included in scoring should be documented when comparing results. The score also does not measure actual viewer engagement.

## Business Applications

The discovered content segments could support further analysis in several areas.

**Catalogue organization:** Content groups may help describe similarities across titles and identify patterns in the catalogue.

**Content discovery:** Interpretable segments could inform experiments with browsing categories or recommendation strategies, subject to separate recommendation-system evaluation.

**Content portfolio analysis:** Catalogue segments may help analysts compare content characteristics across genres, release periods, or other available metadata.

**Engagement research:** If viewing and interaction data become available, content clusters can be compared with watch time, completion rates, repeat viewing, or other engagement measures.

These are potential applications of the analysis rather than demonstrated business outcomes from clustering alone.

## Limitations

- Catalogue metadata does not directly measure whether viewers watched or enjoyed a title.
- Missing values and inconsistent metadata can affect cluster quality.
- Clustering results depend on feature selection, scaling, encoding, and algorithm parameters.
- DBSCAN may classify many observations as noise or produce clusters of very different sizes.
- Silhouette scores can be misleading or difficult to interpret for certain cluster structures, particularly when noise points are present.
- Content similarity does not necessarily imply audience preference or engagement.
- Results from catalogue metadata should not be interpreted as evidence of Netflix's internal recommendation algorithms or business decisions.

## Future Improvements

- Document and validate the preprocessing pipeline and feature selection.
- Examine cluster sizes, noise points, and representative titles from each cluster.
- Test clustering robustness across parameter settings and feature combinations.
- Compare silhouette scores with other suitable clustering evaluation measures.
- Explore text-based representations of titles and descriptions if supported by the project scope.
- Integrate actual viewing or engagement data, where lawfully available, to test whether content segments relate to observed audience behavior.
- Develop and evaluate a separate recommendation or engagement-prediction model if suitable labelled data becomes available.

## How to Run the Project

### Prerequisites

- Python 3.10 or a compatible Python environment.
- Access to the dataset used in the notebook.
- Google Colab or Jupyter Notebook.

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/netflix-content-analysis.git
cd netflix-content-analysis
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Open the notebook in Google Colab or Jupyter Notebook, update the dataset path, and run the cells in order.

Replace `YOUR_USERNAME` and the repository name with your actual GitHub details.

## Suggested Repository Structure

```text
netflix-content-analysis/
├── Netflix_Content_Analysis.ipynb
├── README.md
├── requirements.txt
└── images/
    ├── content_distribution.png
    ├── release_year_trends.png
    └── clustering_comparison.png
```

The `images/` folder is optional and should contain only exported visualizations that are actually generated by the project. Adjust the notebook filename to match the file uploaded to your repository.

## Conclusion

This project applies exploratory data analysis and unsupervised learning to investigate patterns in Netflix's content catalogue. By comparing DBSCAN, K-Means, and Agglomerative Clustering, it evaluates alternative approaches to grouping content based on available metadata.

DBSCAN achieved the highest reported silhouette score of 0.7970 in the current experiments, followed by K-Means at 0.6781 and Agglomerative Clustering at 0.6638. These findings provide a basis for further investigation into catalogue segmentation and content similarity.

Although the analysis offers potential applications for content organization and engagement research, actual engagement prediction requires appropriate viewer-behavior data and separate predictive evaluation. The project provides practical experience in data exploration, feature engineering, unsupervised machine learning, model comparison, and the interpretation of analytical results.

## Author

**Anagha P Nair**

Add your GitHub profile or professional portfolio link here.

## License and Data Attribution

Specify an appropriate licence for your code and provide attribution and usage information for the dataset used. Do not assume the dataset may be redistributed merely because it is publicly accessible.
