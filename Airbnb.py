import warnings
import pandas as pd
import numpy as np
import pymongo
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
st.set_page_config(page_title="Airbnb Analysis", layout="centered")


col1, col2 = st.columns([0.1, 0.9])

with col1:
    st.image("https://cdn.freebiesupply.com/logos/large/2x/airbnb-2-logo-svg-vector.svg", width=80)
with col2:
    st.write("# Airbnb Analysis 📇")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Home', '🗃 Data', 'Trend Analysis(Price)', 'Availabilities Trend', 'Location Based Trends', 'Conclusion'])

with tab1:
    st.write("Welcome to the Airbnb Data Analysis app!")

with tab2:
    data = st.file_uploader('Upload an Airbnb CSV file', type=['csv'])

    if data is not None:
        # Read the uploaded file
        data1 = pd.read_csv(data)
        # Create a DataFrame
        df = pd.DataFrame(data1)
        # Display the DataFrame
        st.write("Preview of the uploaded dataset:")
        st.dataframe(df)
    else:
        st.warning("Please upload a CSV file.")
        df = None  # Initialize df to avoid errors in tab3

with tab3:
    if df is not None:
        required_columns = ['property_type', 'price', 'city', 'first_review_date', 'room_type']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"The dataset is missing required columns: {', '.join(missing_columns)}")
        else:

            cola,colb= st.columns(2)
            with cola:
                st.title("Pricing Trends by Property Type")
                property_price = df.groupby('property_type')['price'].mean().sort_values()

                # Matplotlib visualization
                fig, ax = plt.subplots(figsize=(8, 4))
                property_price.plot(kind='bar', color='skyblue', ax=ax)
                ax.set_title("Average Price by Property Type")
                ax.set_ylabel("Average Price (in $)")
                ax.set_xlabel("Property Type")

                st.pyplot(fig)
                
            with colb:
                # Grouping the data by country and city to get average prices
                country_city_price = df.groupby(['country', 'city'])['price'].mean().reset_index()

                # Create a select box for country selection
                unique_countries = sorted(country_city_price['country'].unique())
                selected_country = st.selectbox("Select a Country", unique_countries)

                # Filter data for the selected country
                filtered_data = country_city_price[country_city_price['country'] == selected_country]

                # Get the top 5 cities by average price
                top_5_cities = filtered_data.nlargest(5, 'price')

                # Visualization
                st.title(f"Top 5 Cities by Average Price in {selected_country}")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(top_5_cities['city'], top_5_cities['price'], color='orange')
                ax.set_title(f"Top 5 Cities by Avg Price in {selected_country}")
                ax.set_ylabel("Average Price (in $)")
                ax.set_xlabel("City")

                # Display the chart
                st.pyplot(fig)

                # Display additional details
                st.write("Top 5 Cities:")
                st.write(top_5_cities[['city', 'price']].reset_index(drop=True))

            colc,cold= st.columns(2)
            
            with colc:
                st.title("Price Distribution by Room Type")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(data=df, x='price', hue='room_type', kde=True, ax=ax, bins=30)
                ax.set_title("Price Distribution by Room Type")
                st.pyplot(fig)

            with cold:
                # Handle invalid dates
                df['first_review_date'] = pd.to_datetime(df['first_review_date'], errors='coerce')
                df = df.dropna(subset=['first_review_date'])
                df['year'] = df['first_review_date'].dt.year

                select_price_trend= st.selectbox('Select the Option',['By Cities', 'By Country'])
                if select_price_trend == 'By Cities':
                    
                    trend = df.groupby(['year', 'city'])['price'].mean().reset_index()
                    # Plot for selected cities
                    st.title("Price Trends Over Years (Selected Cities)")
                    selected_cities = st.multiselect("Select Cities", options=df['city'].unique(), default=df['city'].unique()[:5])
                    filtered_data = trend[trend['city'].isin(selected_cities)]

                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.lineplot(data=filtered_data, x='year', y='price', hue='city', ax=ax)
                    ax.set_title(f"Price Trends for {', '.join(selected_cities)}")
                    st.pyplot(fig)
                elif select_price_trend == 'By Country':
                    country_trend = df.groupby(['year', 'country'])['price'].mean().reset_index()

                    # Plot for countries
                    st.title("Price Trends Over Years by Country")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.lineplot(data=country_trend, x='year', y='price', hue='country', ax=ax)
                    ax.set_title("Price Trends for Countries")
                    st.pyplot(fig)

    else:
        st.warning("Please upload a CSV file in the 'Data' tab.")

with tab4:
    if df is not None:
        cole,colf= st.columns(2)
    
        with cole:
            selected_country = st.selectbox(
                "Select a Country to View Top 5 Cities by Availability", 
                df['country'].unique()
            )
            st.title("Top Cities by Availability in Selected Country")
            
            # Filter data for the selected country and calculate city-wise average availability
            filtered_data = df[df['country'] == selected_country]
            filtered_data = filtered_data[filtered_data['availability_365'] > 0]

            city_availability = filtered_data.groupby('city')['availability_365'].mean().reset_index()
            #city_availability = filtered_data.groupby('city')['availability_365'].sum().reset_index()


            # Sort by availability in ascending order (least availability first) and get top 5
            city_availability = city_availability.sort_values(by='availability_365', ascending=True).head(5)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(city_availability['city'], city_availability['availability_365'], color='red')
            ax.set_title(f"Top 5 Cities by Least Avg Availability in {selected_country}")
            ax.set_ylabel("Average Availability (Days)")
            ax.set_xlabel("City")
            ax.tick_params(axis='x', rotation=45)

            # Display in Streamlit
            st.pyplot(fig)


        with colf:
            st.title("Room Type Availability Trends")
            
            # Filter data for the selected country and calculate room type availability
            filtered_data = df[df['country'] == selected_country]
            room_availability = filtered_data.groupby('room_type')['availability_365'].mean().reset_index()
            room_availability = room_availability.sort_values(by='availability_365', ascending=False)

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(room_availability['room_type'], room_availability['availability_365'], color='blue')
            ax.set_title(f"Room Type Availability in {selected_country}")
            ax.set_ylabel("Average Availability (Days)")
            ax.set_xlabel("Room Type")

            # Display in Streamlit
            st.pyplot(fig)


        colg,colh= st.columns(2)

        with colg:
            
            # Calculate average availability by country
            country_availability = df.groupby('country')['availability_365'].mean().reset_index()
            country_availability = country_availability.sort_values(by='availability_365', ascending=False)

            # Streamlit UI
            st.title("Country-Wise Availability Trends")
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(country_availability['country'], country_availability['availability_365'], color='skyblue')
            ax.set_title("Average Property Availability by Country")
            ax.set_ylabel("Average Availability (Days)")
            ax.set_xlabel("Country")
            ax.tick_params(axis='x', rotation=45)

            # Display in Streamlit
            st.pyplot(fig)
            st.markdown('''---------------''')

        with colh:
            st.title("Property Type Availability Trends")
            selected_country = st.selectbox(
                "Select a Country to View Property Type Availability", 
                df['country'].unique(), 
                key="property_type"
            )

            # Filter data for the selected country and calculate property type availability
            filtered_data = df[df['country'] == selected_country]
            property_availability = filtered_data.groupby('property_type')['availability_365'].mean().reset_index()
            property_availability = property_availability.sort_values(by='availability_365', ascending=False).head(10)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(property_availability['property_type'], property_availability['availability_365'], color='purple')
            ax.set_title(f"Top 10 Property Types by Avg Availability in {selected_country}")
            ax.set_xlabel("Average Availability (Days)")
            ax.set_ylabel("Property Type")

            # Display in Streamlit
            st.pyplot(fig)
        
        
        
        
            st.title("Seasonal Availability Trends")
            selected_country = st.selectbox(
                "Select a Country to View Seasonal Trends", 
                df['country'].unique(), 
                key="seasonal"
            )

            # Extract month from the dataset if not already present
            df['month'] = pd.to_datetime(df['last_review_date']).dt.month

            # Filter data for the selected country and calculate monthly availability
            filtered_data = df[df['country'] == selected_country]
            monthly_availability = filtered_data.groupby('month')['availability_365'].mean().reset_index()

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=monthly_availability, x='month', y='availability_365', marker='o', ax=ax)
            ax.set_title(f"Monthly Availability Trends in {selected_country}")
            ax.set_ylabel("Average Availability (Days)")
            ax.set_xlabel("Month")
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(
                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )

            # Display in Streamlit
            st.pyplot(fig)
        
        colk,coll= st.columns(2)

        with colk:
            st.title("Availability vs Price Analysis")
            selected_country = st.selectbox(
                "Select a Country to Compare Availability and Price", 
                df['country'].unique(), 
                key="availability_price"
            )

            # Filter data for the selected country
            filtered_data = df[df['country'] == selected_country]

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(filtered_data['availability_365'], filtered_data['price'], alpha=0.5, color='coral')
            ax.set_title(f"Availability vs Price in {selected_country}")
            ax.set_xlabel("Availability (Days)")
            ax.set_ylabel("Price (in $)")

            st.pyplot(fig)

        with coll:
            st.title("Yearly Availability Trends")
            df['year'] = pd.to_datetime(df['last_review_date']).dt.year  # Extract year from the dataset

            selected_country = st.selectbox(
                "Select a Country for Yearly Trends", 
                df['country'].unique(), 
                key="yearly_trends"
            )

            # Filter and calculate yearly availability
            filtered_data = df[df['country'] == selected_country]
            yearly_availability = filtered_data.groupby('year')['availability_365'].mean().reset_index()

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(yearly_availability['year'], yearly_availability['availability_365'], marker='o', color='orange')
            ax.set_title(f"Yearly Availability Trends in {selected_country}")
            ax.set_ylabel("Average Availability (Days)")
            ax.set_xlabel("Year")

            st.pyplot(fig)
    else:
        st.warning("Please upload a CSV file in the 'Data' tab.")

with tab5:
    if df is not None:

        colm,coln= st.columns(2)

        with colm:
            st.title("Impact of Amenities on Price")

            # Scatter plot to show the relationship between amenities count and price
            st.subheader("Scatter Plot: Number of Amenities vs. Price")

            # Create the scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='amenities_count', y='price', alpha=0.7, ax=ax, color='teal')
            ax.set_title("Number of Amenities vs. Price")
            ax.set_xlabel("Number of Amenities")
            ax.set_ylabel("Price (in $)")
            st.pyplot(fig)

        with coln:
            st.subheader("Price Distribution by Number of Amenities") #Understand the spread of prices across different levels of amenities.

            # Create bins for amenities count
            df['amenities_bins'] = pd.cut(df['amenities_count'], bins=[0, 10, 20, 30, 40, 50, 60],
                                        labels=["0-10", "11-20", "21-30", "31-40", "41-50", "51-60"])
            
            lower_cap_log = df['price_log'].quantile(0.05)
            upper_cap_log = df['price_log'].quantile(0.95)
            df['price_log'] = df['price_log'].clip(lower=lower_cap_log, upper=upper_cap_log)

            sns.boxplot(x=df['amenities_bins'], y=df['price_log'], ax=ax, palette='Blues')
            ax.set_title("Adjusted Log-Transformed Price Distribution by Number of Amenities")
            ax.set_xlabel("Number of Amenities (Grouped)")
            ax.set_ylabel("Log Price")
            st.pyplot(fig)


        colo,colp= st.columns(2)
        with colo:
            st.subheader("Average Price by Number of Amenities") #Purpose: Identify clear trends in how price changes with amenities.

            # Calculate average price for each bin
            avg_price_by_amenities = df.groupby('amenities_bins')['price'].mean()

            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            avg_price_by_amenities.plot(kind='bar', color='orange', ax=ax)
            ax.set_title("Average Price by Number of Amenities")
            ax.set_xlabel("Number of Amenities (Grouped)")
            ax.set_ylabel("Average Price (in $)")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with colp:
            st.subheader("Heatmap: Amenities Count by Country") #Purpose: Highlight cities where properties offer the most amenities.

            # Pivot table for heatmap
            heatmap_data = df.pivot_table(values='amenities_count', index='country', columns='room_type', aggfunc='mean').fillna(0)

            # Heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".1f", ax=ax)
            ax.set_title("Average Number of Amenities by Country and Room Type")
            ax.set_xlabel("Room Type")
            ax.set_ylabel("Country")
            st.pyplot(fig)

        colq,colr= st.columns(2)
        with colq:
            st.subheader("Distribution of Amenities Count") #Purpose: Identify the most common ranges of amenities.

            # Histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['amenities_count'], bins=20, kde=True, color='green', ax=ax)
            ax.set_title("Distribution of Amenities Count")
            ax.set_xlabel("Number of Amenities")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    else:
        st.warning("Please upload a CSV file in the 'Data' tab.")

with tab6:
    st.markdown('''
- Top Cities by Price:
    - The analysis revealed that cities like [list top cities] exhibit the highest average prices. This suggests that these locations are premium markets with strong demand, likely driven by their desirability or scarcity of listings.
- Price Distribution by Amenities:
    - Properties with more amenities tend to have higher average prices, emphasizing the value of offering quality amenities to attract guests. However, extreme outliers indicate that properties priced excessively high may face challenges in occupancy, suggesting the need for balanced pricing.
- Availability Trends:
    - Listings with high availability often correlate with mid-range pricing, while low-availability properties are priced at a premium. This highlights the importance of understanding seasonal demand and optimizing pricing to maximize occupancy and revenue.
- Location-Based Trends:
    - Certain locations have both high demand and limited availability, making them attractive for new investments or listing expansions. Conversely, areas with high availability but lower prices may indicate oversupply or underutilized potential.
- Price Distribution Patterns:
    - The log-transformed price distribution revealed a right-skewed market with a majority of affordable listings, while a smaller segment targets luxury-seeking travelers. Outliers in this distribution suggest opportunities for better market segmentation.
- Amenities vs. Price:
    - Grouping properties by amenities showed a clear trend: properties offering a balanced set of amenities (e.g., 20–40 items) perform well in terms of pricing and guest attraction. Extremely amenity-rich or amenity-poor properties may not yield proportionate returns.
- Insights from Outliers:
    - The presence of pricing outliers suggests variability in host strategies or guest preferences. Proper filtering and analysis of these cases could provide deeper insights into unique property types or niche markets.
''')