#import libraries 
import streamlit as st 
import pandas as pd
import networkx as nx
from pymed import PubMed
import plotly.graph_objects as go
from itertools import combinations
import base64 

def get_pubmed_articles(query: str, max_results: int = 50) -> list:
    pubmed = PubMed(tool="MyTool", email="my@email.address")
    # Convert the query to lowercase
    query = query.lower()
    results = pubmed.query(query, max_results=max_results)
    articles = []
    for article in results:
        articleDict = article.toDict()
        articles.append(articleDict)
    return articles

def create_dataframe(articles: list) -> pd.DataFrame:
    articleInfo = []
    for article in articles:
        pubmedId = article['pubmed_id'].split(',')[0].strip()
        articleInfo.append({
            'pubmed_id': pubmedId,
            'title': article.get('title', ''),
            'keywords': article.get('keywords', []),
            'journal': article.get('journal', ''),
            'abstract': article.get('abstract', ''),
            'conclusions': article.get('conclusions', ''),
            'methods': article.get('methods', ''),
            'results': article.get('results', ''),
            'copyrights': article.get('copyrights', ''),
            'doi': article.get('doi', ''),
            'publication_date': article.get('publication_date', ''),
            'authors': article.get('authors', [])
        })
    return pd.DataFrame.from_dict(articleInfo)

def get_table_download_link(df, filename="data.csv"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="pubmed_articles.csv">Download CSV File</a>'
    return href

def extract_keywords(articles):
    all_keywords = []
    for article in articles:
        keywords = [keyword.lower() for keyword in article.get('keywords', [])]
        all_keywords.extend(keywords)
    return list(set(all_keywords))  # Remove duplicates

def create_co_occurrence_network(articles, keywords):
    G = nx.Graph()
    G.add_nodes_from(keywords)
    
    # Dictionary to store co-occurrences
    co_occurrences = {}
    
    for article in articles:
        article_keywords = sorted([keyword.lower() for keyword in article.get('keywords', [])])
        for kw1, kw2 in combinations(article_keywords, 2):
            if kw1 in keywords and kw2 in keywords:
                # Ensure the pair is always in the same order
                pair = tuple(sorted([kw1, kw2]))
                if pair in co_occurrences:
                    co_occurrences[pair] += 1
                else:
                    co_occurrences[pair] = 1
    
    # Add edges to the graph
    for (kw1, kw2), weight in co_occurrences.items():
        G.add_edge(kw1, kw2, weight=weight)
    
    # Create DataFrame
    df = pd.DataFrame([(kw1, kw2, weight) for (kw1, kw2), weight in co_occurrences.items()],
                      columns=['Keyword1', 'Keyword2', 'Co-occurrences'])
    
    # Sort DataFrame by co-occurrences in descending order
    df = df.sort_values('Co-occurrences', ascending=False).reset_index(drop=True)
    
    return G, df

def plot_interactive_co_occurrence_network(G):
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        ),
        text=[],
        textposition="top center"
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f"{adjacencies[0]}: {len(adjacencies[1])} connections")

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Keyword Co-occurrence Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    return fig


# Main app
if __name__ == "__main__":
    st.title("PubMed Article CSV & Keyword Co-Occurrence Analysis")
    st.write("Create a dataframe with most recent articles on desired topics")
    st.write("Generate a co-occurrence analysis to help guide your search")

    # Search input
    query = st.text_input("Enter your search query:", "")

    if query:
        articles = get_pubmed_articles(query)
        
        if articles:
            # Create dataframe
            df_articles = create_dataframe(articles)
            
            # Display download link for articles
            st.markdown(get_table_download_link(df_articles, "pubmed_articles.csv"), unsafe_allow_html=True)

            # Keyword co-occurrence analysis
            st.subheader("Keyword Co-occurrence Analysis")
            keywords = extract_keywords(articles)
            if len(keywords) > 1:
                co_occurrence_network, df_co_occurrences = create_co_occurrence_network(articles, keywords)
                
                # Display interactive network graph
                fig = plot_interactive_co_occurrence_network(co_occurrence_network)
                st.plotly_chart(fig)
                
                # Display co-occurrence DataFrame
                st.subheader("Keyword Co-occurrence Table")
                st.dataframe(df_co_occurrences)
                
                # Download link for co-occurrence data
                st.markdown(get_table_download_link(df_co_occurrences, "keyword_co_occurrences.csv"), unsafe_allow_html=True)
                
                # Additional analysis options
                st.subheader("Keyword Analysis")
                selected_keyword = st.selectbox("Select a keyword for detailed analysis:", keywords)
                if selected_keyword:
                    total_co_occurrences = df_co_occurrences[
                        (df_co_occurrences['Keyword1'] == selected_keyword) | 
                        (df_co_occurrences['Keyword2'] == selected_keyword)
                    ]['Co-occurrences'].sum()
                    st.write(f"Total co-occurrences for '{selected_keyword}': {total_co_occurrences}")
                    
                    # Display top co-occurring keywords
                    st.write(f"Top keywords co-occurring with '{selected_keyword}':")
                    top_co_occurring = df_co_occurrences[
                        (df_co_occurrences['Keyword1'] == selected_keyword) | 
                        (df_co_occurrences['Keyword2'] == selected_keyword)
                    ].sort_values('Co-occurrences', ascending=False).head(10)
                    st.table(top_co_occurring)
            else:
                st.write("Not enough keywords found for co-occurrence analysis.")
        else:
            st.write("No articles found for the given query.")
    else:
        st.write("Enter a search query to get started.")