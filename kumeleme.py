from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn.impute import SimpleImputer
import pandas as pd


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)




def kumele():
    veri = pd.read_csv("data.csv", sep=',')

    imp = SimpleImputer(missing_values=-12345, strategy='mean')
    # clears all columns and be ready to be processed!
    veriyeni = imp.fit_transform(veri)

    inputt = veriyeni[:, 1:12]

    n_clusters = 4
    sample_size = 20
    n_features = 12
    init_value = 10

    print(82 * '_')


    bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=init_value),
                  name="k-means++", data=inputt)

    pca = PCA(n_components=n_clusters).fit(inputt)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=init_value),
                  name="PCA-based",
                  data=inputt)
    print(82 * '_')

    # #############################################################################
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=2).fit_transform(inputt)
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=init_value)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=8)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    lastkmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=init_value)
    lastkmeans.fit(inputt)

    numaralar = []
    satirlar = []

    for val in inputt:
        numaralar.append(lastkmeans.predict([val, val])[1])
        # print("{} şeklindeki veri satırının ait olduğu küme numarası {} olarak belirlenmiştir.".format(val,lastkmeans.predict([val, val])[1]))

    adict = {}

    for i in range(n_clusters):
        adict[str(i)] = 1

    for i in numaralar:
        adict[str(i)] = adict[str(i)] + 1

    enCokKullanilanKume = -1
    enCokKumeElemani = 0

    for i in range(n_clusters):
        if(adict[str(i)] > enCokKumeElemani):
            enCokKullanilanKume = i
            enCokKumeElemani = adict[str(i)]


    benzerSatirlar = []

    for val in inputt:
        if(lastkmeans.predict([val, val])[1] == enCokKullanilanKume):
            benzerSatirlar.append(val)

    enCokKullanilanOzellikler = np.zeros(11)


    for satir in benzerSatirlar:
        print(satir)
        for j in range(len(satir)):
            if(satir[j] == 1):
                enCokKullanilanOzellikler[j] += 1


    yedek = enCokKullanilanOzellikler.copy()

    yedek = sorted(yedek)

    son = yedek[-1]
    sondanbir = yedek[-2]


    enCokKullanilanIndis = 0
    enCokKullanilanIndis2 = 0

    for i in range(len(enCokKullanilanOzellikler)):
        if(enCokKullanilanOzellikler[i] == son):
            enCokKullanilanIndis = i

        elif(enCokKullanilanOzellikler[i] == sondanbir):
            enCokKullanilanIndis2 = i

    return((enCokKullanilanIndis, enCokKullanilanIndis2))
