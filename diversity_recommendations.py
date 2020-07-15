import csv
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle as pkl

'''
    Inputs:
        - daily stock returns for 10 years.
    Outputs:
        - saves preprocessed data, with nulls removed. Ready for PCA.
'''
def input_df_from_csv():
    stocks_df = pd.read_csv('data/stocks_daily_raw_10yr.csv')
    print("input data head:")
    print(stocks_df.head())

    #filter inputs & recreate index
    stocks_df_5yr = stocks_df[1511:]
    stocks_df_5yr = stocks_df_5yr.reset_index(drop=True)
    print("\n=================\nfiltered (x yr) data head:")
    print(stocks_df_5yr.head())


    #filter out the nulls
    stocks_ser = stocks_df_5yr.isnull().sum()
    stocks_ser_nonnulls = stocks_ser[stocks_ser <= 20]
    nnullkeys = stocks_ser_nonnulls.keys()
    print("\nNon-null keys: ", nnullkeys)
    nnull_stocks_df = stocks_df_5yr[nnullkeys]
    #print(nnull_stocks_df.head())

    #convert NaN's to 0.
    #"X" should be ready for pca
    X = nnull_stocks_df.iloc[:,1:-1]
    X = X.fillna(0)

    print(X.head())

    #perform cumulative product:
    X_cp = (1+X).cumprod()
    print("\n=================\ncomputing cumulative returns:")
    print(X_cp.tail())
    X = X_cp

    #do a null check: if nulls, throw error?
    print("\n=================\ncheckign for nulls:")
    nullsums = X.isnull().sum()
    print(nullsums[nullsums >0])

    #store X in a csv
    X.to_csv('data/input_stocks_data_X.csv')


'''
    Inputs: stock data X
    Outputs: take the input X, perform pca, and create eigenportfolios
'''
def find_eigen_portfolios(colnames, pcs):
    epfs = []
    epfs_inv = []

    for i in range(len(pcs)):
        #want to inspect & create cutoffs.
        ci = pcs[i]

        #debug
        #print(len(ci))

        maxi = max(ci)
        mini = min(ci)
        print("***** starting with pc #{}".format(i))
        print("max: ", max(ci), "min: ", min(ci))

        #find # correlations greater than half of max, or less than half of min.
        csig_pos = np.arange(len(ci))[ci > maxi/2]
        csig_neg = np.arange(len(ci))[ci < mini/2]
        print("significance lengths")
        print(len(csig_pos))
        print(len(csig_neg))

        #create lists of tickers.
        csig = csig_pos if len(csig_pos) >= len(csig_neg) else csig_neg
        csig_inv = csig_neg if len(csig_pos) >= len(csig_neg) else csig_pos
        ptiks = [colnames[i] for i in csig]
        ptiks_inv = [colnames[i] for i in csig_inv]

        epfs.append(ptiks)
        epfs_inv.append(ptiks_inv)
        print(ptiks[:20])

    with open('data/pcs/eigenportfolios.pkl', 'wb') as fout:
        pkl.dump(epfs, fout)
    with open('data/pcs/eigenportfolios_inv.pkl', 'wb') as fout:
        pkl.dump(epfs_inv, fout)

def compute_pca_eigen_portfolios():
    X = pd.read_csv('data/input_stocks_data_X.csv')
    colnames = list(X.columns)
    print(colnames)
    #scale X
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(X)
    #perform PCA
    print("\n=================\nPerforming PCA...")
    pca = PCA(n_components=5)
    Xpca = pca.fit_transform(scaled_x)
    print("transformed data (Xpca) shape & values:")
    print(Xpca.shape)
    print(Xpca)

    #results analysis:
    #print the variance ratio
    print("\n\nexplained variance: ", pca.explained_variance_ratio_)
    #print the singular values
    print("\nsingular values: ", pca.singular_values_)
    #print pca components and their shape:
    print("\ncomponents shape & values: ", pca.components_.shape)
    print(pca.components_)

    #save pc's in a file.
    pc_df = pd.DataFrame(data=pca.components_)
    with open('data/pcs/principal_components.csv', 'w') as fout:
        pc_df.to_csv(fout)

    #next, come up with the eigen portfolios as identified by the PC's
    find_eigen_portfolios(colnames, pca.components_)


'''
    Input: Principal components
    Outputs: eigenpf recommendation, & weight of eigenpf's
'''
def diversity_recommendation():
    #first, identify the factor loadings of the portfolio stocks in each principal component.
    pcs = pd.read_csv('data/pcs/principal_components.csv')
    print(pcs.head())
    np_pcs = pcs.iloc[:,1:].to_numpy()

    #find all the right factor loadings in this result.
    portfolio = ['AAPL', 'AMZN', 'MSFT', 'INTC']

    X = pd.read_csv('data/input_stocks_data_X.csv')
    colnames = list(X.columns)
    #find index of all portfolio elements
    pf_inds = [colnames.index(p) for p in portfolio]
    print(pf_inds)

    #find the factor loadings of all pcs. Then do a squared sum.
    facloads = np_pcs[:, pf_inds]
    print("factor loadings: ", facloads)
    variances = (facloads**2).sum(1) #squares & sums across rows
    norm_vcs = variances / (sum(variances))
    print("\n")
    print("normalized var's in all pc's: ", norm_vcs)

    #print diff between correspondence in this pf, and overall market. Capture the max diff.
    mkt_vars = [0.48753289, 0.1872592,  0.07927921, 0.04877565, 0.03629214]
        ## TODO: NEED TO DO THIS^^ FOR THE COMPUTED PCA VARIANCES.
    missing_var = mkt_vars-norm_vcs
    print("where is pf variance behind market: ", missing_var)

    ### == return this result! == ###
    print("eigen pf to invest in: ", np.argmax(missing_var))
    print("eigen pf diversity weight: ", np.max(missing_var))
    return np.argmax(missing_var), np.max(missing_var)



#----------------------------------------
#           run the above.
#----------------------------------------

#run the preprocessing
input_df_from_csv()
#run eigenportfolio creation.
compute_pca_eigen_portfolios()
#create diversity recommendations.
epf_ind, missing_var = diversity_recommendation()


#next, run the nearest neighbors with this input, to get the final rec.
