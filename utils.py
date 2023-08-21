import numpy as np
import pandas as pd
import re
import time
import joblib
import itertools
import collections
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, LearningCurveDisplay
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# @contextmanager
# def timer(title):
#     t0 = time.time()
#     yield
#     print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', ' ', x))
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('Data/application_train.csv', nrows= num_rows)
    df_test = pd.read_csv('Data/application_test.csv')
    print("Train size: {}".format(len(df)))
    print("Test size: {}".format(len(df_test)))
    df = df.append(df_test).reset_index()

    # Optional: Remove 4 applications with XNA CODE_GENDER
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('Data/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('Data/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('Data/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev

    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('Data/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos

    return pos_agg

# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('Data/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins

    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('Data/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc

    return cc_agg

# Display/plot feature importance
def display_importances(feature_importance_df_, title='LightGBM Features (avg over folds)', display=True, save=True):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig('Data/features_importances.png')
    if display:
        plt.show()
    plt.close()

# Display/plot the ROC curve and the AUC
def plot_roc_auc(y_test, y_pred_proba, display=True, save=True):  
    
    # Compute roc curve et auc score
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # get the best threshold
    label, counts = np.unique(y_test, return_counts=True)
    r = counts[0]/counts[1]
    J = tpr - (r/10)*fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    
    plt.figure(figsize=(8,6))
    plt.plot(
        fpr,
        tpr,
        lw=0.5,
        marker='.',
        markersize=0.5,
        color="darkorange",
        label="ROC curve (area = {})".format(round(auc_score, 2))
    )
    plt.plot([0, 1], [0, 1],
             color="navy", lw=2, linestyle="--", label='No Skill')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', 
                 label='Best thresholds = {}'.format(round(best_thresh, 3)))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save:    
        plt.savefig('Data/roc_auc_curve.png')
    if display:
        plt.show()
    plt.close()
    
    return best_thresh

# Display/plot confusion matrix
def plot_confusion_matrix(cm, target_names, title, 
                          normalize=True, save_path='Data/matrix.png',
                          display=True, save=True):
    
    # accuracy = np.trace(cm) / float(np.sum(cm))
    # misclass = 1 - accuracy
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0])
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(5,5))
    plt.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm_norm.max() / 1.5

    # if normalize:
    plt.text(0, 0, "{:,}\n(TN)".format( cm[0, 0]),
             horizontalalignment="center",
             color="white" if cm_norm[0, 0] > thresh else "black")
    plt.text(1, 0, "{:,}\n(FP)".format(cm[0, 1]),
             horizontalalignment="center",
             color="white" if cm_norm[0, 1] > thresh else "black")
    plt.text(0, 1, "{:,}\n(FN)".format(cm[1, 0]),
             horizontalalignment="center",
             color="white" if cm_norm[1, 0] > thresh else "black")
    plt.text(1, 1, "{:,}\n(TP)".format(cm[1, 1]),
             horizontalalignment="center",
             color="white" if cm_norm[1, 1] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nRecall={:0.4f}; FalsePositiveRate={:0.4f}'\
               .format(recall, fpr))
    
    plt.gcf().canvas.draw()
    if save:
        plt.savefig(save_path, bbox_inches="tight")
    if display:
        plt.show()
    plt.close()

# Compute and Display/plot Learning curve
def plot_learning_curve(X_train, y_train, estimator, display=True, save=True):
    
    fig, ax = plt.subplots(figsize=(10, 6))

    common_params = {
        "X": X_train,
        "y": y_train,
        "train_sizes": np.linspace(0.02, 1.0, 10),
        "cv": 5,
        "score_type": "both",
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "roc_auc",
    }


    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax)
    handles, label = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Training Score", "Test Score"])
    ax.set_title("Learning Curve for Logistic Regression")
    ax.set_ylim(0.50, 0.8)
    plt.tight_layout()
    if save:
        plt.savefig('Data/learning_curve.png')
    if display:
        plt.show()
    plt.close()

# Display/plot probas distributions
def plot_hist_proba(X, y, th, display=True, save=True):
    
    d = {}
    d['proba'] = X
    d['true_class'] = y
    data = pd.DataFrame(d)
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    sns.histplot(ax=ax, data=data, x='proba', hue='true_class', kde=True)
    h1, l1 = ax.get_legend().legendHandles, ['0', '1']

    y = ax.get_ylim()
    plt.plot([0.5, 0.5], y,
                 color="navy", lw=2, linestyle="--", 
                 label='Threshold (0.5)')
    plt.plot([th, th], y,
                 color="r", lw=2, linestyle="--", 
                 label='Best Threshold ({})'.format(round(th, 3)))

    ax.add_artist(ax.legend(h1, l1, loc='center right'))
    ax.legend()
    ax.set_title("Distribution des probas prédites")
    plt.tight_layout()
    if save:
        plt.savefig('Data/distribution_pred_probas.png')
    if display:
        plt.show()
    plt.close()

# Fonctions pour clustering client
def clustering_scores(X, clusters, display=False):
    """
    Compute scores for clustering :
        - Distortion
        - Calinsky Harabasz
        - Silhouette
        - Gini
    -----------------------------------
    Params:
        X : pd.Dataframe : samples utilisées pour le clustering (sans les labels des clusters)
        clusters : pd.series ou array (mm longueur que X) : labels de nos samples
        display : boolean : print ou non les résultats
    Returns:
        results : array : array contenant nos scores
    """
    
    # Calcul des coordonées des clusters
    
    centers = [] # liste d'array , coordonnées des clusters
    k = len(clusters.unique()) # nombre de cluster
    
    for i in sorted(clusters.unique(), reverse=True):
        
        center = X.loc[clusters == i, :].mean().values
        centers.append(center)
        
    if k == 1:
        return([0,0,0,0])
       
        
    
    # Score de Distortion
    
    distortion = 0
    for i in range(X.shape[0]):
        distortion += min([metrics.pairwise.euclidean_distances(
                            X.iloc[i].to_numpy().reshape(1, -1), c.reshape(1, -1))**2 for c in centers]).item()
    
    # Score Calinsky Harabasz
    
    cali = metrics.calinski_harabasz_score(X, clusters)
        
    # Score Silhouette    
        
    sil = metrics.silhouette_score(X, clusters)
    
    # Score de Gini
    
    gin = gini(clusters)
        
    # Affichage des resultats
    
    if display:
        
        print('Score avec {} clusters: '.format(k))
        print('Score de Distortion : {:,}'.format(round(distortion)))
        print('Score de Calinsky Harabasz : {:,}'.format(round(cali)))
        print('Score de Silhouette : {}'.format(round(sil, 2)))
        print('Score de Gini : {}'.format(round(gin, 3)))
        
    
    return([distortion, cali, sil, gin])
    
# =======================================================================================================

# Gap Statistic for K means
    
def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic 
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
    # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
    # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):

            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp
        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)


    return (gaps.argmax() + 1, resultsdf)

# =======================================================================================================

# Gini coefficient
    
def gini(clusters_labels):
    """Compute the Gini coefficient for a clustering.
    Parameters:
    - clusters_labels: pd.Series of labels of clusters for each point.
    """

    # Get frequencies from clusters_labels
    clusters_labels = pd.Series(clusters_labels)
    frequencies = clusters_labels.value_counts()

    # Mean absolute difference
    mad = frequencies.mad()

    # Mean frequency of clusters
    mean = frequencies.mean()

    # Gini coefficient
    gini_coeff = 0.5 * mad / mean

    return gini_coeff

# =======================================================================================================

def kmeans_function(data, scaler=None, K=None):
    """Fonction pour clustering K-means :
        - Scaling des données
        - Visualisations de différentes staistiques (Distortion, Silhouette, Calinsky et Gap)
        - Run K-means avec le nombre optimal de cluster défini par le score de Distortion
        - Renvoie un dataset avec une nouvelle colonne contenant les labels des clusters
        
    -----------------------------------
    data : pd.DataFrame : données que l'on veut "cluster"
    scaler : sklearn.preprocessing : scaler à appliquer sur data avant kmeans   
    """
    
    if scaler is not None:
        X_scaled = scaler.fit_transform(data) # Scaling des données
    else:
        X_scaled = data.copy()
        
    
    
    model = KMeans() # On instancie kmean
    
    # Création de la figure pour visualiser les scores
    
    plt.figure(figsize=(24, 14))
        
    ax1 = plt.subplot(2,3,1) # Distortion Score Elbow
    ax2 = plt.subplot(2,3,2) # Silhouette Score Elbow
    ax3 = plt.subplot(2,3,3) # Calinsky Harabasz Score Elbow
    ax4 = plt.subplot(2,2,3) # Silhouette plot
    ax5 = plt.subplot(2,2,4) # Gap statistic
    
    visualizer_1 = KElbowVisualizer(model, k=(2,8), timings= True, ax=ax1) # k : nb de clusters
    visualizer_1.fit(X_scaled)        # Fit sur nos données
    visualizer_1.finalize()
    
    if K is None:
        K = visualizer_1.elbow_value_    # On récupère la valeur opti de cluster selon le score de distortion
    
    visualizer_2 = KElbowVisualizer(model, k=(2,8), metric='silhouette', timings= True, ax=ax2) 
    visualizer_2.fit(X_scaled) 
    visualizer_2.finalize()
        
    visualizer_3 = KElbowVisualizer(model, k=(2,8), metric='calinski_harabasz', timings= True, ax=ax3) 
    visualizer_3.fit(X_scaled)        
    visualizer_3.finalize()
    
    opt_model = KMeans(K)  # On instancie kmean avec notre k optimal (selon le score de distortion)
    visualizer_4 = SilhouetteVisualizer(opt_model, ax=ax4)
    visualizer_4.fit(X_scaled)    
    visualizer_4.finalize()    
    
    score_g, df_g = optimalK(X_scaled, nrefs=5, maxClusters=8)
    ax5.plot(df_g['clusterCount'], df_g['gap'], linestyle='--', marker='o', color='b')
    ax5.set_xlabel('K')
    ax5.set_ylabel('Gap Statistic')
    ax5.set_title('Gap Statistic vs. K')
    
    plt.show()
    
    # Implémentation d'une colonne cluster, contenant les labels des clusters, dans la dataframe initiale
    
    start = time.time()               # Fit le modèle avec k opti 
    kmeans = opt_model.fit(X_scaled)  # et mesure le temps de calcul
    end = time.time()
    delta = round((end - start), 3)
    
    r_data = data.copy()
    r_data['cluster'] = kmeans.labels_
    
    # Récupération des centres de chaque cluster
    
    if scaler is not None:
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
    else:
        centers = kmeans.cluster_centers_
        
    centers = pd.DataFrame(centers, columns=data.columns)
    
    counter = collections.Counter(kmeans.labels_) 
    centers['nb_clients'] = list(dict(sorted(counter.items())).values())
    
    # Affichage du temps de calcul de l'algorithme et de l'indice de Gini
    
    gin = gini(r_data['cluster'])
    
    print('Temps d\'exécution avec k = {} : {}s'.format(K, delta))
    print('Indice de Gini : {}'.format(gin))
    
    return(r_data, centers)
