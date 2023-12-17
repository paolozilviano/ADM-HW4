import numpy as np
from collections import Counter
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# We want to gather the title and genre of the top 10 movies that each user clicked on regarding the number of clicks.
def user_gather(df):
    
    # First we create a column with the clicks count of each film made by each user and we order the dataset by user ids and clicks numbers
    df['click_count'] = df.groupby(['user_id', 'title'])['Unnamed: 0'].transform('count')
    df_sorted = df.sort_values(by=['user_id', 'click_count'], ascending=[True, False])
    
    # Then we select the first 10 values from each user (to select only the 10 movies most clicked by each user)
    # dropping the duplicates and we group only the columns we need to work with by the user id
    df_top10 = df_sorted[['user_id', 'title', 'genres', 'click_count']].drop_duplicates().groupby('user_id').head(10)
    user_data = df_top10.groupby(by='user_id')
    
    # Now we can gather the title and genre of the top 10 movies from each user
    user_movies_genres = {}
    for key, item in user_data:
    
        # Creating a set containing the top 10 movies for the specific user
        movie_set = set(item.title.values)
    
        # Creating a set containing the genres associated to the top 10 movies from the specific user
        genres_set = set()
        for el in item.genres:
            genres_set = genres_set | set(el.split(', '))
        if 'NOT AVAILABLE' in genres_set:
            genres_set.remove('NOT AVAILABLE')
    
        # We can fill the dictionary with the user_id as key and as value a list cointaining the two sets created above (top 10 movies and genres for each user)
        user_movies_genres[item.user_id.iloc[0]] = [movie_set, genres_set]
        
    return user_movies_genres, df_top10


# We need the set of all the genres present in the dataset so we take the unique value present in the genre column in the dataset
def genres_set(df):
    genres_set = set()

    for el in df.genres:
        for genre in el.split(', '):
            genres_set.add(genre)
        
    genres_set.remove('NOT AVAILABLE')
    return genres_set


# In this function we generet a matrix representation of the genres and movies sets
def generate_sets_matrix(genres_set, user_movies_genres):
    
    # We starts creating a matrix with n° genres rows and n° user columns filled with zeros.
    matrix_sets = np.zeros((len(genres_set), len(user_movies_genres)))
    
    # Creating a user ids list wich contains all the user in the dataset and a genres list wich cointains all the genres in the dataset
    user_ids_list = list(user_movies_genres.keys())
    genres_list = list(genres_set)
    
    # We set each column value equal to 1 if the user's top 10 genres contrains that genre
    for i in range (matrix_sets.shape[1]):
        for j in range (matrix_sets.shape[0]):
            if genres_list[j] in user_movies_genres[user_ids_list[i]][1]:
                    matrix_sets[j][i] = 1
    return matrix_sets, user_ids_list


# We define out hash function that we'll use in the next function in ordere to generate the signature matrix
def hash_func (k, x):
    return (k**2 * x + 7) % 47


# In this function we'll generate the signature matrix
def generate_sign_matrix(matrix_sets):
    
    # First we create a matrix with n (90 in this case) rows wich means that we'll use n hash functions too (a different hash function for each rows).
    sign_matrix = np.full((90, matrix_sets.shape[1]), 99999)
    
    for i in range (matrix_sets.shape[0]):
        
        # For each hash function we calcultate the n-hash value based on the row number and we put it in am hash-value list.
        hash_values = []
        for k in range (sign_matrix.shape[0]):
            k_hash_values = hash_func(k, i + 1)
            hash_values.append(k_hash_values)
            
        # For each column (so we can analize each value in the row we're analizing) if value == 1 we check it's associated hash value
        for j in range (matrix_sets.shape[1]):
            
            if matrix_sets[i][j] == 1:
                # For each hash value if i-hash value is lower than the one in the signature matrix we change it
                for k in range (sign_matrix.shape[0]):
                    if hash_values[k] < sign_matrix[k][j]:
                        sign_matrix[k][j] = hash_values[k]
    return sign_matrix
    

# In this function we'll create b new buckets which means that we will consider any titles with the same first (sign_matrix.shape[1]/b) rows to be similar. 
def generate_buckets(r, sign_matrix):
    buckets = []
    k = 0
    
    # We create a bucket for each band 
    for j in range(0, sign_matrix.shape[0], r):
        buckets.append({})
    
        # For each band we group users together in different sets if they have the same values in the fifteen values that we're checking
        for i in range (sign_matrix.shape[1]):
            hash_vec = ''
            for n in range (j, j + r):
                hash_vec += str(sign_matrix[n][i])
            if hash_vec not in buckets[k].keys():
                buckets[k][hash_vec] = {i}
            elif hash_vec in buckets[k].keys():
                pv = buckets[k][hash_vec]
                pv.add(i)
                buckets[k][hash_vec] = pv
        k += 1
    return buckets


# In this function we create a list of dictionary (one for each band) with the user_id as key and as value a set cointaining all the other similar users
def generate_vocabulary(buckets, user_ids_list):
    user_list = [{} for _ in range(6)]
    i = 0
    
    for bucket_band in buckets:
        keys = bucket_band.keys()
        
        for bucket in keys:
            if len(bucket_band[bucket]) > 1:
                user_set = bucket_band[bucket]
                
                for user in user_set:
                    if user_ids_list[user] not in user_list[i].keys():
                        user_list[i][user_ids_list[user]] = user_set
        i += 1
    return user_list

# Gievn an user_id first of all we need to find all the similar users.
# We first join all the users associated to the user_id given in a list and take the most common ones.
def find_similar_users(user_list, user_ids_list, query):
    sim_user_list = []
    for i in range(len(user_list)):
        if query in user_list[i].keys():
            sim_user_list += list(user_list[i][query])
    
    # Then we count the most similar user based on how many time they're in the same band buckets and we select the 3 most similar users
    # (because each user is equal to itself and we need to delete it if necessary). So at the end we'll have the 2 most similar users.
    count_freq = Counter(sim_user_list).most_common(3)
    i = 0
    final_user_list = []
    for el in count_freq:
        if user_ids_list[el[0]] != query and i < 2:
            final_user_list.append(user_ids_list[el[0]])
            i += 1
    return final_user_list

# This function will return the ordered 5 recommended movies for the given user
def recommended_movies(user_movies_genres, df_top10, final_user_list):
    
    # First we save the movies watched by the 2 most simil users in two sets and we look for any movies in common
    movies_1 = user_movies_genres[final_user_list[0]][0].copy()
    movies_2 = user_movies_genres[final_user_list[1]][0].copy()
    common_movies = movies_1.intersection(movies_2)
    
    # Then we filter the dataset selecting only the data for the selected users
    # We'll have a dataset for each user containing only the title of the top ten movies watched and the click counts.
    dati_utente_1 = df_top10[df_top10['user_id'] == final_user_list[0]]
    dati_utente_1 = dati_utente_1[['title', 'click_count']]
    dati_utente_2 = df_top10[df_top10['user_id'] == final_user_list[1]]
    dati_utente_2 = dati_utente_2[['title', 'click_count']]
    
    # Then we create a list (sorted_movies) wich will contain all the movies required.
    # First we check the common movies set and we order the movies by the sum cof the clicks number
    comm_movies_dict = dict()
    for el in common_movies:
        comm_movies_dict[el] = dati_utente_1[dati_utente_1['title'] == el].click_count.iloc[0] + dati_utente_2[dati_utente_2['title'] == el].click_count.iloc[0]
        movies_1.remove(el)
        movies_2.remove(el)
    sorted_movies = sorted(comm_movies_dict, key=lambda x: x)
    
    # Then eventually we check the rest of the top-10 movies for the most similar user
    if len(sorted_movies) < 5:
        movies_1_dict = dict()
        for el in movies_1:
            movies_1_dict[el] = dati_utente_1[dati_utente_1['title'] == el].click_count.iloc[0]
        sorted_movies_1 = sorted(movies_1_dict, key=lambda x: x)
        sorted_movies += sorted_movies_1
    
    # Then eventually we check the rest of the top-10 movies for the second most similar user
    if len(sorted_movies) < 5:
        movies_2_dict = dict()
        for el in movies_2:
            movies_2_dict[el] = dati_utente_2[dati_utente_2['title'] == el].click_count.iloc[0]
        sorted_movies_2 = sorted(movies_2_dict, key=lambda x: x)
        sorted_movies += sorted_movies_2

    return sorted_movies[:5]

# We created a function to calculate time until midnight
# for durations spanning next day
def calculate_time_until_midnight(row):
    if row['datetime'].hour * 3600 + row['datetime'].minute * 60 + row['duration'] > 86400:  # 86400 seconds in a day
        return (pd.Timestamp('2023-01-01 23:59:59') - row['datetime']).seconds
    else:
        return row['duration']

#K-means clustering algorithm
def init_centroids(data, k):
    # Initialize centroids randomly by selecting k data points.
    indices = np.random.choice(len(data), size=k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    # Assign data points to the nearest centroid.
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, cluster_labels, k):
    # Update centroids based on the mean of data points in each cluster.
    return np.array([data[cluster_labels == ci].mean(axis=0) for ci in range(k)])

def k_means(data, k=3, max_iter=100):
    # Initialize centroids using random data points
    centroids = init_centroids(data, k)

    # Iterate until convergence or maximum iterations
    for _ in range(max_iter):
        cluster_labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, cluster_labels, k)

        # Check for convergence by comparing centroids
        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    # Calculate the Sum of Squared Errors (SSE)
    sse = sum(np.linalg.norm(data[cluster_labels == ci] - centroids[ci]) ** 2 for ci in range(k))

    return centroids, cluster_labels, sse

def elbow_method(data, k_range):
    # Initialize an empty list to store SSE values
    sse_values = []

    # Iterate over different values of K
    for k in tqdm(k_range):
        try:
            # Perform K-means clustering for the current K
            _, _, sse = k_means(data, k)
            
            sse_values.append(sse)
            print(f"K={k}, SSE={sse}")
        except Exception as e:
            # Handle any errors during clustering
            print(f"An error occurred at K={k}: {e}")

    # Return the list of SSE values for each K
    return sse_values

def silhouette_scores_method(data, k_range):
    # Initialize an empty list to store Silhouette Scores
    silhouette_scores = []

    # Iterate over different values of K
    for k in tqdm(k_range):
        # Perform K-means clustering for the current K
        _, cluster_labels, _ = k_means(data, k)
        
        print('Model is done')
        score = silhouette_score(data, cluster_labels)
        
        print('silhouette_score is done')
        silhouette_scores.append(score)
        
        # Print the Silhouette Score for the current K
        print(f"K={k}, Silhouette Score={score}")

    # Return the list of Silhouette Scores for each K
    return silhouette_scores

# K-means++ clustering algorithm
def initialize_plusplus_centroids(data, k):
    """
    - data: Input data array with shape (n_samples, n_features)
    - k: Number of clusters

    Returns 'centroids': Initial centroids array with shape (k, n_features)
    """
    num_samples, num_features = data.shape
    centroids = np.empty((k, num_features), dtype=data.dtype)
    centroids[0] = data[np.random.randint(num_samples)]

    for i in range(1, k):
        # Calculate squared distances from each data point to the nearest existing centroid
        squared_distances = np.min(np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :i, :]) ** 2, axis=2), axis=1)
        # Calculate probabilities for each data point to be selected as the next centroid
        probabilities = squared_distances / np.sum(squared_distances)
        cumulative_probabilities = np.cumsum(probabilities)
        # Randomly choose the next centroid based on the calculated probabilities
        random_value = np.random.rand()
        centroid_index = np.searchsorted(cumulative_probabilities, random_value)
        centroids[i] = data[centroid_index]

    return centroids

def k_means_plusplus(data, num_clusters=3, max_iterations=100):
    """
    - data: Input data array with shape (n_samples, n_features).
    - num_clusters: Nr of clusters.
    - max_iterations: Max nr of iterations.

    Returns:
    - centroids: Final centroids array with shape (num_clusters, n_features).
    - cluster_labels: Array of assigned cluster labels for each data point.
    - sse
    """
    # Initialize centroids using KMeans++
    initial_centroids = initialize_plusplus_centroids(data, num_clusters)

    for _ in range(max_iterations):
        # Assign data points to clusters based on the current centroids
        cluster_labels = np.argmin(np.linalg.norm(data[:, np.newaxis, :] - initial_centroids, axis=2), axis=1)

        # Update centroids based on the mean of data points in each cluster
        new_centroids = np.array([data[cluster_labels == i].mean(axis=0) for i in range(num_clusters)])
        # Check for convergence by comparing centroids
        if np.array_equal(initial_centroids, new_centroids):
            break

        initial_centroids = new_centroids

    # Calculate Sum of Squared Errors 
    sse = np.sum((data - initial_centroids[cluster_labels]) ** 2)
    return initial_centroids, cluster_labels, sse
