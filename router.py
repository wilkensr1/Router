import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
import googlemaps
import os
import numpy as np
import time
from sklearn.cluster import DBSCAN
import folium

# Debugging prints
print("Current Working Directory:", os.getcwd())
print("Contents of Directory:", os.listdir())

# Read CSV file
def read_csv(file_path):
    df = pd.read_csv(file_path)
    print("CSV Loaded. Number of records:", len(df))
    return df

# Geocode addresses with retries
def geocode_address(address, max_retries=3):
    geolocator = Nominatim(user_agent="route_optimizer")
    gmaps = googlemaps.Client(key=[Google API Key]')

    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(address)
            if location:
                return (location.latitude, location.longitude)
        except Exception as e:
            print(f"Geocoding error for address {address}: {e}")
        time.sleep(1)  # Wait a bit before retrying
    # Fallback to Google Maps Geocoding API
    try:
        gmaps_result = gmaps.geocode(address)
        if gmaps_result:
            location = gmaps_result[0]['geometry']['location']
            return (location['lat'], location['lng'])
    except Exception as e:
        print(f"Google Maps geocoding error for address {address}: {e}")
    return (None, None)

# Convert addresses to coordinates
def convert_addresses_to_coords(df):
    coords = []
    failed_addresses = []
    for index, row in df.iterrows():
        address = f"{row['AddressLine1']}, {row['City']}, {row['State']} {row['ZipCode']}"
        lat, lon = geocode_address(address)
        if lat and lon:
            coords.append((lat, lon))
        else:
            failed_addresses.append(address)
            print(f"Failed to geocode address: {address}")
    print("Total geocoded addresses:", len(coords))
    print("Total failed addresses:", len(failed_addresses))
    return coords

# Filter out NaN values
def filter_valid_coords(coords):
    valid_coords = [coord for coord in coords if not any(pd.isna(c) for c in coord)]
    return valid_coords

# Manually balance clusters
def balance_clusters(coords, num_teams):
    coords = np.array(coords)
    kmeans = KMeans(n_clusters=num_teams, random_state=0).fit(coords)
    labels = kmeans.labels_

    while True:
        cluster_sizes = [list(labels).count(i) for i in range(num_teams)]
        max_size = max(cluster_sizes)
        min_size = min(cluster_sizes)
        
        if max_size - min_size <= 1:
            break
        
        largest_cluster = cluster_sizes.index(max_size)
        smallest_cluster = cluster_sizes.index(min_size)
        
        # Find a point to move from the largest cluster to the smallest
        point_to_move = None
        for i, label in enumerate(labels):
            if label == largest_cluster:
                point_to_move = i
                break
        
        if point_to_move is not None:
            labels[point_to_move] = smallest_cluster
    
    return labels

# Optimize routes (using Google Maps API)
def optimize_routes(clusters, coords):
    gmaps = googlemaps.Client(key='AIzaSyDQNm3hFsmbSz44zF8mh7kKl4LIy6QuBPA')
    routes = []
    for cluster_id in set(clusters):
        cluster_coords = [coords[i] for i in range(len(coords)) if clusters[i] == cluster_id]
        
        if cluster_coords:
            # Extracting and formatting coordinates for the API call
            waypoints = [(coord[0], coord[1]) for coord in cluster_coords]
            
            # Request directions with optimized waypoints
            response = gmaps.directions(
                origin=waypoints[0],
                destination=waypoints[-1],
                waypoints=waypoints[1:-1],
                mode="driving",
                optimize_waypoints=True
            )
            routes.append(response)
    return routes

# Write output to CSV with team-specific sections
def write_routes_to_csv(routes, output_path, df, clusters):
    # Initialize a dictionary to store team routes
    team_routes = {i: [] for i in set(clusters)}
    
    # Extract names and addresses for each team
    for cluster_id in set(clusters):
        cluster_indices = [i for i in range(len(clusters)) if clusters[i] == cluster_id]
        for index in cluster_indices:
            row = df.iloc[index]
            name = row['AthleteFirstName']
            address = f"{row['AddressLine1']}, {row['City']}, {row['State']}, {row['ZipCode']}"
            team_routes[cluster_id].append((name, address))
    
    # Write team-specific sections to CSV
    with open(output_path, 'w') as f:
        for team_id, addresses in team_routes.items():
            f.write(f"Team {team_id}\n")
            unique_addresses = set(addresses)  # Ensure unique addresses
            for addr in unique_addresses:
                f.write(f"{addr}\n")
            f.write("\n")

# Plot addresses on map
def plot_addresses_on_map(df, clusters, output_map_path):
    team_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    my_map = folium.Map(location=map_center, zoom_start=12)

    for cluster_id in set(clusters):
        cluster_indices = [i for i in range(len(clusters)) if clusters[i] == cluster_id]
        for index in cluster_indices:
            row = df.iloc[index]
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"{row['AthleteFirstName']}<br>{row['AddressLine1']}<br>{row['City']}, {row['State']} {row['ZipCode']}",
                icon=folium.Icon(color=team_colors[cluster_id % len(team_colors)])
            ).add_to(my_map)
    
    my_map.save(output_map_path)

# Main function
def main(file_path, num_teams, output_path):
    df = read_csv(file_path)
    coords = convert_addresses_to_coords(df)
    valid_coords = filter_valid_coords(coords)

    if len(valid_coords) < num_teams:
        raise ValueError("Number of teams exceeds number of valid coordinates.")
    
    # Add coordinates to the DataFrame
    df = df.iloc[:len(valid_coords)].copy()
    df['Latitude'] = [coord[0] for coord in valid_coords]
    df['Longitude'] = [coord[1] for coord in valid_coords]

    
    print("Valid coordinates:", len(valid_coords))
    if len(valid_coords) < num_teams:
        raise ValueError("Number of teams exceeds number of valid coordinates.")
    
    clusters = balance_clusters(valid_coords, num_teams)
    routes = optimize_routes(clusters, valid_coords)
    write_routes_to_csv(routes, output_path, df, clusters)
    plot_addresses_on_map(df, clusters, output_map_path)

# Example usage
file_path = 'athletes.csv'  # Ensure this path is correct
num_teams = 4
output_path = 'optimized_routes.csv'
output_map_path = 'teams_map.html'

try:
    main(file_path, num_teams, output_path)
except ValueError as ve:
    print(f"Error: {ve}")
