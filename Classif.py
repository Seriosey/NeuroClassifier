# coding=windows-1251
import pandas as pd
from pandas import HDFStore
import numpy as np
import math
import random as rn
import h5py
import matplotlib.pyplot as plt
import pickle
row = 350


def create_train():
    train1 = [[0.35, 1.571, 24, 11, 8.2, 5.6, 122, 32], 
          [0.35, 3.141, 29.8, 12.6, 10.4, 5.4, 25.4, 7],
          [0.35, 0, 28.8, 4.2, 22.8, 22.8, 4.8, 3.5],
          [0.35, 3.141, 27.1, 9.1, 15, 1, 110, 16],
          [0.35, -1.571, 4, 1.3, 3, 3.6, 5.2, 6.9],
          [0.35, -1.571, 8.8, 3.3, 1.2, 1.4, 1.7, 2],
          [0.35, 0, 8, 2, 5, 2, 5, 2]]

    train_feat = np.array(train1[:])
    train_data = train_feat[:, [0, 1, 2, 4, 6]]

    pvbas = []
    pvbas_phi = []
    pvbas_R = []
    pvbas_theta_sr = []
    pvbas_nontheta_sr = []
    pvbas_ripple_sr = []

    for i in range(21):
        pvbas_R.append(rn.normalvariate(0.35, 0.05))
        pvbas_phi.append(rn.normalvariate(1.57, 0.837))
        pvbas_theta_sr.append(rn.normalvariate(24, 11))
        pvbas_nontheta_sr.append(rn.normalvariate(8.2, 5.6))
        pvbas_ripple_sr.append(rn.normalvariate(122, 32))
    pvbas.extend([pvbas_R, pvbas_phi, pvbas_theta_sr, pvbas_nontheta_sr, pvbas_ripple_sr])
    pvbas = np.array(pvbas)
    pvbas = pvbas.transpose()


    olm = []
    olm_phi = []
    olm_R = []
    olm_theta_sr = []
    olm_nontheta_sr = []
    olm_ripple_sr = []

    for i in range(21):
        olm_R.append(rn.normalvariate(0.35, 0.05))
        olm_phi.append(rn.normalvariate(3.14, 0.55))
        olm_theta_sr.append(rn.normalvariate(29.8, 12.6))
        olm_nontheta_sr.append(rn.normalvariate(10.4, 5.4))
        olm_ripple_sr.append(rn.normalvariate(25.4, 7))
    olm.extend([olm_R, olm_phi, olm_theta_sr, olm_nontheta_sr, olm_ripple_sr])
    olm = np.array(olm)
    olm = olm.transpose()


    aac = []
    aac_phi = []
    aac_R = []
    aac_theta_sr = []
    aac_nontheta_sr = []
    aac_ripple_sr = []

    for i in range(21):
        aac_R.append(rn.normalvariate(0.35, 0.05))
        aac_phi.append(rn.normalvariate(0, 0.49))
        aac_theta_sr.append(rn.normalvariate(28.8, 4.2))
        aac_nontheta_sr.append(rn.normalvariate(22.8, 5.4))
        aac_ripple_sr.append(rn.normalvariate(4.8, 3.5))
    aac.extend([aac_R, aac_phi, aac_theta_sr, aac_nontheta_sr, aac_ripple_sr])
    aac = np.array(aac)
    aac = aac.transpose()


    bis = []
    bis_phi = []
    bis_R = []
    bis_theta_sr = []
    bis_nontheta_sr = []
    bis_ripple_sr = []

    for i in range(20):
        bis_R.append(rn.normalvariate(0.35, 0.05))
        bis_phi.append(rn.normalvariate(3.14, 0.29))
        bis_theta_sr.append(rn.normalvariate(27.1, 9.1))
        bis_nontheta_sr.append(rn.normalvariate(15, 1))
        bis_ripple_sr.append(rn.normalvariate(110, 16))
    bis.extend([bis_R, bis_phi, bis_theta_sr, bis_nontheta_sr, bis_ripple_sr])
    bis = np.array(bis)
    bis = bis.transpose()


    ivy = []
    ivy_phi = []
    ivy_R = []
    ivy_theta_sr = []
    ivy_nontheta_sr = []
    ivy_ripple_sr = []

    for i in range(20):
        ivy_R.append(rn.normalvariate(0.35, 0.05))
        ivy_phi.append(rn.normalvariate(-1.57, 0.64))
        ivy_theta_sr.append(rn.normalvariate(4, 1.3))
        ivy_nontheta_sr.append(rn.normalvariate(3, 3.6))
        ivy_ripple_sr.append(rn.normalvariate(5.2, 6.9))
    ivy.extend([ivy_R, ivy_phi, ivy_theta_sr, ivy_nontheta_sr, ivy_ripple_sr])
    ivy = np.array(ivy)
    ivy = ivy.transpose()


    cckbas = []
    cckbas_phi = []
    cckbas_R = []
    cckbas_theta_sr = []
    cckbas_nontheta_sr = []
    cckbas_ripple_sr = []

    for i in range(20):
        cckbas_R.append(rn.normalvariate(0.35, 0.05))
        cckbas_phi.append(rn.normalvariate(-1.57, 1.4))
        cckbas_theta_sr.append(rn.normalvariate(8.8, 3.3))
        cckbas_nontheta_sr.append(rn.normalvariate(1.2, 1.4))
        cckbas_ripple_sr.append(rn.normalvariate(1.7, 2))
    cckbas.extend([cckbas_R, cckbas_phi, cckbas_theta_sr, cckbas_nontheta_sr, cckbas_ripple_sr])
    cckbas = np.array(cckbas)
    cckbas = cckbas.transpose()


    ngl = []
    ngl_phi = []
    ngl_R = []
    ngl_theta_sr = []
    ngl_nontheta_sr = []
    ngl_ripple_sr = []

    for i in range(20):
        ngl_R.append(rn.normalvariate(0.35, 0.05))
        ngl_phi.append(rn.normalvariate(0, 0.76))
        ngl_theta_sr.append(rn.normalvariate(8, 2))
        ngl_nontheta_sr.append(rn.normalvariate(5, 2))
        ngl_ripple_sr.append(rn.normalvariate(5, 2))
    ngl.extend([ngl_R, ngl_phi, ngl_theta_sr, ngl_nontheta_sr, ngl_ripple_sr])
    ngl = np.array(ngl)
    ngl = ngl.transpose()


    train_data = np.concatenate ((train_data, pvbas, olm, aac, bis, ivy, cckbas, ngl), axis= 0 )
    return  train_data


def reform_data(_path):

    table = pd.read_hdf(_path)
    #table = pd.read_hdf('feasures_table.hdf5')
    print(table) #C:\Users\serio\Source\Repos\InterneuronsProject
    #print('\n\n\n')

    n1 = np.array(table[:])#Copying hdf5 data in numpy array

    #print(n1)
    #print('\n\n\n')

    row = 350  #Took only 150 test neurons to decrease size of print output
    col = 5 

    train_data = create_train()

    test_data = n1[:, [1, 0, 16, 18, 20]]# Choosing columns with theta_R, theta_phi, ts_spike_rate_std, non_ts_spike_rate_std, ripples_spike_rate_std

    #Concatenating train and test data for mutual preprocessing
    data = np.concatenate ((train_data, test_data), axis= 0 )

    #print('data in descartes:')
    for i in range(row):
        tR = data[i][0]
        data[i][0] = tR*math.cos(data[i][1])
        data[i][1] = tR*math.sin(data[i][1])
        #transfering first two columns from polar to descart

        import scipy.stats as stats

    #zscoring every column and creating zscored matrix 'preproc_data'

    phi_x = np.empty(row)
    for i in range (row):
        phi_x[i] =  data[i][0]
    zsc_x = (phi_x - phi_x.mean())/phi_x.std()
 
    phi_y = np.empty(row)
    for i in range (row):
        phi_y[i] = data[i][1]
    zsc_y = (phi_y - phi_y.mean())/phi_y.std()

    theta = []
    for i in range (row):
        theta.append(data[i][2])
    zsc_theta = stats.zscore(theta)

    nontheta = []
    for i in range (row):
        nontheta.append(data[i][3])
    zsc_nontheta = stats.zscore(nontheta)

    ripple = []
    for i in range (row):
        ripple.append(data[i][4])
    zsc_ripple = stats.zscore(ripple)


    preproc_data = [] 
    preproc_data.extend([zsc_x, zsc_y, zsc_theta, zsc_nontheta, zsc_ripple]) 

    f_data = np.array(preproc_data)
    f_data = f_data.transpose()
    
    return f_data

#for i in range(row):  
#   for j in range(5):  
#      print(data[i][j], end = " ")
#   print()
#print('\n\n\n')


#types = ['PVBAS','OLM', 'AAC', 'BIS', 'Ivy', 'CCKBAS', 'NGF' ]

def create_true_labels():
    labels = np.array([[1], [2], [3], [4], [5], [6], [7]])

    for i in range(21):
        labels = np.append(labels, [1])

    for i in range(21):
        labels = np.append(labels, [2])

    for i in range(21):
        labels = np.append(labels, [3])

    for i in range(20):
        labels = np.append(labels, [4])

    for i in range(20):
        labels = np.append(labels, [5])

    for i in range(20):
        labels = np.append(labels, [6])

    for i in range(20):
        labels = np.append(labels, [7])

    return labels


# Numbers in respect with types above
#labels = labels.transpose()
#print(labels)
#print(len(labels))
#print(len(fin_train_data))
#print(len(labels)==len(fin_train_data))

'''from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 3 )

model.fit(fin_train_data, labels.ravel())

predictions = model.predict(fin_test_data)'''


'''from sklearn.ensemble import RandomForestClassifier

forest_model = RandomForestClassifier(max_depth=2, random_state=0)
forest_model.fit(fin_train_data, labels.ravel())
predictions2 = forest_model.predict(fin_test_data)'''

#print(model.predict([[ 0.26429427,  1.50245036,  0.01001849, -0.77408003,  0.68862798]]))


def indents_hist():
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    indents = []
    for i in range(150):
        #with np.set_printoptions(precision=3, suppress = True):

            x = label_prop_model.predict_proba([fin_test_data[i,:]])
            #print(x)
            s = np.sort(x[0])
            print(s[6]-s[5])
            indents.append(s[6]-s[5])

    plt.hist(indents, density = True, color = 'blue', edgecolor = 'black', bins = int(1/0.01))


def getR_gamma(fin_test_data):
    predictions1 = classify(fin_test_data)

    gamma_s = n1[:, 3]
    gamma_m = n1[:, 5] 
    gamma_f = n1[:, 7]

    ngl_indexes, pvbas_indexes, olm_indexes, aac_indexes, bis_indexes, ivy_indexes, cck_indexes = ([] for i in range (7))
    ngl_gamma_s, pvbas_gamma_s, olm_gamma_s, aac_gamma_s, bis_gamma_s, ivy_gamma_s, cck_gamma_s = ([]for i in range (7))
    ngl_gamma_m, pvbas_gamma_m, olm_gamma_m, aac_gamma_m, bis_gamma_m, ivy_gamma_m, cck_gamma_m = ([] for i in range (7))
    ngl_gamma_f, pvbas_gamma_f, olm_gamma_f, aac_gamma_f, bis_gamma_f, ivy_gamma_f, cck_gamma_f = ([] for i in range (7))

    for i in range(len(predictions1)):
        if predictions1[i] == 7: ngl_indexes.append(i)
        if predictions1[i] == 1: pvbas_indexes.append(i)
        if predictions1[i] == 2: olm_indexes.append(i)
        if predictions1[i] == 3: aac_indexes.append(i)
        if predictions1[i] == 4: bis_indexes.append(i)
        if predictions1[i] == 5: ivy_indexes.append(i)
        if predictions1[i] == 6: cck_indexes.append(i)

    for i in ngl_indexes:
        ngl_gamma_s.append(gamma_s[i])
        ngl_gamma_m.append(gamma_m[i])
        ngl_gamma_f.append(gamma_f[i])

    for i in pvbas_indexes:
        pvbas_gamma_s.append(gamma_s[i])
        pvbas_gamma_m.append(gamma_m[i])
        pvbas_gamma_f.append(gamma_f[i])

    for i in olm_indexes:
        olm_gamma_s.append(gamma_s[i])
        olm_gamma_m.append(gamma_m[i])
        olm_gamma_f.append(gamma_f[i])

    for i in aac_indexes:
        aac_gamma_s.append(gamma_s[i])
        aac_gamma_m.append(gamma_m[i])
        aac_gamma_f.append(gamma_f[i])

    for i in bis_indexes:
        bis_gamma_s.append(gamma_s[i])
        bis_gamma_m.append(gamma_m[i])
        bis_gamma_f.append(gamma_f[i])

    for i in ivy_indexes:
        ivy_gamma_s.append(gamma_s[i])
        ivy_gamma_m.append(gamma_m[i])
        ivy_gamma_f.append(gamma_f[i])

    for i in cck_indexes:
        cck_gamma_s.append(gamma_s[i])
        cck_gamma_m.append(gamma_m[i])
        cck_gamma_f.append(gamma_f[i])

    

    mean_ngl_gamma_s = np.mean(ngl_gamma_s)
    mean_ngl_gamma_m = np.mean(ngl_gamma_m)
    mean_ngl_gamma_f = np.mean(ngl_gamma_f)

    print(np.mean(ngl_gamma_s),'\n',np.mean(ngl_gamma_m),'\n', np.mean(ngl_gamma_f))
    print('\n\n\n')
    print(np.mean(pvbas_gamma_s),'\n',np.mean(pvbas_gamma_m),'\n', np.mean(pvbas_gamma_f))
    print('\n\n\n')
    print(np.mean(olm_gamma_s),'\n',np.mean(olm_gamma_m),'\n', np.mean(olm_gamma_f))
    print('\n\n\n')
    print(np.mean(aac_gamma_s),'\n',np.mean(aac_gamma_m),'\n', np.mean(aac_gamma_f))
    print('\n\n\n')
    print(np.mean(bis_gamma_s),'\n',np.mean(bis_gamma_m),'\n', np.mean(bis_gamma_f))
    print('\n\n\n')
    print(np.mean(ivy_gamma_s),'\n',np.mean(ivy_gamma_m),'\n', np.mean(ivy_gamma_f))
    print('\n\n\n')
    print(np.mean(cck_gamma_s),'\n',np.mean(cck_gamma_m),'\n', np.mean(cck_gamma_f))
    

def scatter(i, j):
    if (i<5&j<5&i>=0&j>=0):
        predictions1 = classify(fin_test_data)
        #scatter = plt.scatter(fin_test_data[:,i], fin_test_data[:, j], c=predictions1)
        scatter = plt.scatter(data[150:300,i], data[150:300, j], c=predictions1)


#figure, axis = plt.subplots(5, 5)
  
## For Sine Function
#axis[0, 0].plot.scatter(data[150:300,0], data[150:300, 1], c=predictions1)
#axis[0, 0].set_title("x-y")
  
## For Cosine Function
#axis[0, 1].plt.scatter(data[150:300,0], data[150:300, 2], c=predictions1)
#axis[0, 1].set_title("x-theta_sr")
  
## For Tangent Function
#axis[1, 0].plt.scatter(data[150:300,0], data[150:300, 3], c=predictions1)
#axis[1, 0].set_title("Tangent Function")
  
## For Tanh Function
#axis[1, 1].plt.scatter(data[150:300,0], data[150:300, 4], c=predictions1)
#axis[1, 1].set_title("Tanh Function")

#xlist=[] 
#ylist = []
#thetalist = []
#non_thetalist = []
#ripplelist = []
#for x in fin_test_data[:,0]: xlist.append(x)
#for y in fin_test_data[:,1]: ylist.append(y)
#for theta in fin_test_data[:,2]: thetalist.append(theta)
#for non_theta in fin_test_data[:,3]: non_thetalist.append(non_theta)
#for ripple in fin_test_data[:,1]: ripplelist.append(ripple)


def scatterPCA(_path):
        from sklearn.decomposition import PCA

        f_data = reform_data(_path)
        fin_test_data = f_data[150:]

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(fin_test_data)

        with open('model.pkl', 'rb') as f:
            label_prop_model = pickle.load(f)

        new_data = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])[['PC1', 'PC2']]

        ['PVBAS','OLM', 'AAC', 'BIS', 'Ivy', 'CCKBAS', 'NGF' ]
        predictions1 = label_prop_model.predict(fin_test_data)
        colors = []
        for x in predictions1: 
            if x == 1: colors.append('red')
            if x == 2: colors.append('orange')
            if x == 3: colors.append('yellow')
            if x == 4: colors.append('green')
            if x == 5: colors.append('lightblue')
            if x == 6: colors.append('darkblue')
            if x == 7: colors.append('purple')

        plt.scatter(new_data['PC1'], new_data['PC2'], c=colors)

        plt.xlabel('PC1')
        plt.ylabel('PC2')

        plt.show()



def do(_path):
    

    f_data = reform_data(_path)

    fin_train_data = f_data[:150] #Spliting data into train and test
    fin_test_data = f_data[150:]

    #import sklearn
    #from sklearn.semi_supervised import LabelPropagation
    #label_prop_model = LabelPropagation(kernel='rbf')
    

    #labels = create_true_labels()
    #label_prop_model.fit(fin_train_data, labels.ravel())

    #with open('model.pkl','wb') as f:
    #    pickle.dump(label_prop_model, f)

    with open('model.pkl', 'rb') as f:
            label_prop_model = pickle.load(f)

    predictions1 = label_prop_model.predict(fin_test_data)

    type_ratio = np.zeros(7)
    for x in predictions1:
        if x == 1: type_ratio[0]+=1
        if x == 2: type_ratio[1]+=1
        if x == 3: type_ratio[2]+=1
        if x == 4: type_ratio[3]+=1
        if x == 5: type_ratio[4]+=1
        if x == 6: type_ratio[5]+=1
        if x == 7: type_ratio[6]+=1

    for i in range(len(type_ratio)): type_ratio[i]/=len(predictions1)

    print(type_ratio)

    

    



    return type_ratio

    

scatterPCA('C:\Users\serio\Downloads\feasures_table0.hdf5')
