# from: http://nbviewer.jupyter.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb
import numpy as np
import random
random.seed(1)
import math
try:
    import matplotlib.pyplot as plt
except:
    print('cant plot. install matplotlib if you want to visualize')
import pickle

#[' BMI', ' BMI Percentile', ' Fundal H', ' HC', ' HC Percentile', ' H', ' Ht Percentile', ' Pre-gravid W', ' W', ' Wt Change', ' Wt Percentile', Temp, BP, HR, Wt for Length Percentile]
def construct_temporal_data(xtrain, headers, ytrain, ylabels, mux=[], stdx=[], subset=np.array([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False])):
    # print(xtrain.shape)
    # print('temporal headers', headers)
    time_steps = 10
    newh = np.array(headers).reshape(time_steps, len(subset))
    newh = np.array(newh)[:, subset]
    newx = xtrain.reshape(xtrain.shape[0], time_steps, len(subset))
    newx = newx[:, :, subset]
    if (len(mux) > 0) and (len(stdx) > 0):
        muxreshape = mux.reshape(time_steps, len(subset))[:,subset]
        stdxreshape = stdx.reshape(time_steps, len(subset))[:,subset]

    pickle.dump(obj=(newx[:,1:,:], newh[1:,:], ytrain, ylabels, muxreshape[1:,:], stdxreshape[1:,:]), file=open('tmpobj_20170811.pkl', 'wb'), protocol=2)
    return newx[:,1:,:], newh[1:,:], muxreshape[1:,:], stdxreshape[1:,:]

def unpickle_data(fname='tmpobj_20170828.pkl'):
    (newx, newh, ytrain, ylabels, mux, stdx) = pickle.load(open(fname, 'rb'))
    return (newx, newh, ytrain, ylabels, mux, stdx)

def euclid_dist(t1,t2):
    return np.sqrt(sum((t1-t2)**2))

def euclid_dist_w_missing(t1, t2):
    dist = 0
    dist_cnt = 0
    # print(t1.shape, t2.shape)
    for i in range(len(t1)):
        nonzeros = (t1[i,:] != 0) & (t2[i,:] != 0)
        if nonzeros.sum() != 0:
            dist += np.sqrt(sum((t1[i,nonzeros]-t2[i, nonzeros])**2)) / nonzeros.sum()
            dist_cnt += 1

    # import pdb
    # pdb.set_trace()

    return dist/dist_cnt if dist_cnt>0 else float('inf')

def DTWDistance(s1, s2, w):
    # nonzeros = (s1 != 0) & (s2 != 0)
    # t1, t2 = s1[nonzeros], s2[nonzeros]

    # print(s1.shape, len(s1), w)
    
    DTW,DTW_cnt={},{}

    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
            DTW_cnt[(i, j)] = 0

    DTW[(-1, -1)] = 0
    
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            nonzero = (s1[i,:] != 0) & (s2[j,:] !=0)
            if nonzero.sum() != 0:
                dist = np.sqrt(sum((s1[i,nonzero]-s2[j,nonzero])**2)) / nonzero.sum()
                items = np.array([DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)]])
                itemdir = items.argmin()
                itemdist = DTW_cnt[(i-1, j)] if (itemdir == 0) else DTW_cnt[(i, j-1)] if (itemdir == 1) else DTW_cnt[(i-1, j-1)]
                DTW[(i, j)] = dist + items.min()
                DTW_cnt[(i, j)] = 1 + itemdist 
            else:
                items = np.array([DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)]])
                itemdir = items.argmin()
                itemdist = DTW_cnt[(i-1, j)] if (itemdir == 0) else DTW_cnt[(i, j-1)] if (itemdir == 1) else DTW_cnt[(i-1, j-1)]
                DTW[(i, j)] = items.min()
                DTW_cnt[(i, j)] = itemdist

    # import pdb
    # pdb.set_trace()
    return DTW[len(s1)-1, len(s2)-1]/DTW_cnt[len(s1)-1, len(s2)-1] if (DTW_cnt[len(s1)-1, len(s2)-1]!=0) else  float('inf')

def k_means_clust(data, num_clust, num_iter, headers, centroids=None, cluster_again=True, distType='euclidean', cross_valid=True):
    if cluster_again == True:
        print('clustering begins. this will take a few minutes depending on iterations:', num_iter, ' and number of clusters:', num_clust, ' and distance type:', distType)
    else:
        cross_valid = False
        print('not recomputing new clusters but rather assigning data points to the existing clusters.')
    
    if cross_valid == True:
        hyperparameterlist = [(8, 100, 'euclidean'), (16, 100, 'euclidean')]#, (4, 100, 'euclidean'),(8, 100, 'dtw'), (16, 100, 'dtw'),(4, 100, 'dtw')]
        best_dist = float('inf')
        best_hyppar = (0,0)
        for (nmcl, nitr, distypei) in hyperparameterlist:
            ix_shuffle = np.arange(0, len(data))
            np.random.shuffle(ix_shuffle)
            twothirds = int(2*len(data)/3)
            datatrain = data[ix_shuffle[0:twothirds], :, :]
            datatest = data[ix_shuffle[twothirds:], :, :]
            centroidstrain, assignmentstrain, trendVarstrain, standardDevCentroidstrain, cnt_clusterstrain, distancestrain = k_means_clust(datatrain, nmcl, nitr, headers, centroids, cluster_again, distypei, cross_valid=False)
            centroidstest, assignmentstest, trendVarstest, standardDevCentroidstest, cnt_clusterstest, distancestest = k_means_clust(datatest, nmcl, nitr, headers, centroidstrain, cluster_again=False, distType=distypei, cross_valid=False)
            print(' test distance is : {0:4.3f}'.format(distancestest.mean()))
            if distancestest.mean() < best_dist:
                best_dist = distancestest.mean() 
                best_hyppar = (nmcl, nitr, distypei)
            if best_dist == float('inf'):
                print('ERROR: Your data probably has a lot of missing values!')
        return k_means_clust(data, best_hyppar[0], best_hyppar[1], headers, centroids, cluster_again, best_hyppar[2], cross_valid=False)
    
    if centroids == None:
        centroids = random.sample(list(data), num_clust)
    
    counter = 0
    trendVars = None
    if cluster_again == False:
        num_iter = 1

    assignments, distances, trendVars, standardDevCentroids = None, None, None, None
    for n in range(num_iter):
        trendVars = np.zeros((data.shape[0], num_clust), dtype=float)
        counter+=1
        assignments={}
        distances = np.zeros((data.shape[0]), dtype=float)
        #assign data points to clusters
        for ind, i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind, j in enumerate(centroids):
                # print(euclid_dist_w_missing(i, j), DTWDistance(i, j, 2))
                if distType == 'euclidean':
                    cur_dist = euclid_dist_w_missing(i, j)
                else:
                    cur_dist= DTWDistance(i, j, 2) 
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = c_ind

                if closest_clust == None:
                    closest_clust = 0
            trendVars[ind, closest_clust] = 1 # 2 - min_dist
            distances[ind] = min_dist

            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[ind]
    
        #recalculate centroids of clusters and update avg within-cluster distances
        standardDevCentroids = np.zeros((num_clust, data[0].shape[0], data[0].shape[1]), dtype=float)
        for key in assignments:
            # import pdb
            # pdb.set_trace()
            clust_sum = np.zeros(data[0].shape, dtype=float)
            clust_cnt = np.zeros(data[0].shape, dtype=float)

            for k in assignments[key]:
                clust_sum += data[k]
                clust_cnt += (data[k] != 0) * 1
            clust_cnt[clust_cnt == 0] = 1
            if cluster_again == True:
                centroids[key] = clust_sum / clust_cnt 

            if n == num_iter - 1:
                clust_sum=np.zeros(data[0].shape, dtype=float)
                clust_cnt=np.zeros(data[0].shape, dtype=float)
                for k in assignments[key]:
                    nonzeroix = ((data[k]!=0) & (centroids[key] != 0)) * 1.0
                    clust_sum += (nonzeroix * data[k] - nonzeroix * centroids[key]) ** 2
                    clust_cnt += nonzeroix
                clust_cnt[clust_cnt == 0] = 1.0
                standardDevCentroids[key][:,:] = clust_sum / clust_cnt 

    cnt_clusters = [len(assignments[k]) for k in assignments]
    print("Done! Number of datapoints per cluster is ", cnt_clusters)
    print('average distances is: {0:4.3f}'.format(distances.mean()))
    return centroids, assignments, trendVars, standardDevCentroids, cnt_clusters, distances

def plot_trends(centroids, headers, standardDevCentroids=[], cnt_clusters=[], mux=[], stdx=[]):
    vital_types = [h.strip('-avg0to1').split(':')[1] for h in headers[0,:]]
    print(vital_types)
    sizex = int(math.ceil(np.sqrt(len(centroids))))
    sizey = int(math.ceil(np.sqrt(len(centroids))))
    fig, axes = plt.subplots(sizex, sizey)
    for i in range(0, sizex):
        for j in range(0, sizey):
            centroids_ix =  i*sizey + j
            if centroids_ix >= len(centroids):
                break
            for vitalix in range(0, len(vital_types)): #np.array(subset).nonzero()[0]:
                coefmu = 0
                coefstd = 1
                if len(mux) != 0 and len(stdx) != 0:
                    coefmu = mux[0:4,vitalix]
                    coefstd = stdx[0:4,vitalix]
                axes[i,j].plot(centroids[centroids_ix][0:4,vitalix] * coefstd + coefmu, label=vital_types[vitalix])
                axes[i,j].set_title('Trend:'+str(centroids_ix) + (' cnt:' + str(cnt_clusters[centroids_ix] if cnt_clusters != [] else '')), fontsize=6)
                if len(standardDevCentroids) != 0:
                    axes[i,j].fill_between(range(len(centroids[centroids_ix][0:4,vitalix])), (centroids[centroids_ix][0:4,vitalix]+standardDevCentroids[centroids_ix][0:4,vitalix])* coefstd + coefmu, (centroids[centroids_ix][0:4,vitalix]-standardDevCentroids[centroids_ix][0:4,vitalix])* coefstd + coefmu, alpha=0.1)
    axes[0,0].legend(fontsize = 6)

    
                

def build_endtoend_model(x, h, y, ylabels, xtest, ytest, num_clust=144, num_iter=100):
    try:
        import tensorflow as tf
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        tf.reset_default_graph()
    except:
        print ('tensorflow not loading. Make sure it is installed and can be imported')
        return

    plt.ion()
    # x = x[:, :, subset]
    # xtest = xtest[:, : , subset]
    # h_subset = np.array(h)[:, subset]
    initialized_patterns = random.sample(list(x), num_clust)
    x_input = tf.placeholder(tf.float64, shape=[None, x[0].shape[0], x[0].shape[1]])
    keep_prob = tf.placeholder(tf.float64)
    print('initiating network with:', num_clust, ' clusters')
    patterns = []
    for i in range(0, num_clust): #tf.zeros([x[0].shape[0], x[0].shape[1]])
        patterns.append( tf.Variable(initialized_patterns[i], name='pattern'+str(i)) )
    
    cluster = []
    net_pattern = []
    for p in patterns:
        net_pattern.append(tf.reduce_mean(tf.squared_difference(x_input, tf.nn.dropout(p, keep_prob))))

    net_d = tf.stack(net_pattern)
    loss = tf.reduce_min(net_d)
    cluster_assignment = tf.argmin(net_d)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    sess = tf.InteractiveSession()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    averages = []
    assignments = np.zeros((len(x)), dtype=int)
    patterns_numpy = 0
    for ep in range(0, num_iter):
        losslist = []
        randomix = np.arange(0, len(x))
        np.random.shuffle(randomix)
        assignments = np.zeros((len(x)), dtype=int)
        for i in range(0,len(x)):
            out = sess.run([train_step, loss, cluster_assignment, patterns], feed_dict={x_input: x[randomix[i]].reshape(-1, x[randomix[i]].shape[0], x[randomix[i]].shape[1]), keep_prob:1})
            assignments[randomix[i]] = int(out[2])
            if i == len(x) - 1:
                patterns_numpy = out[3]
                losslist.append(out[1])
                print(len(patterns_numpy), patterns_numpy[0].shape, h_subset.shape)
                plot_trends(patterns_numpy, h); plt.show()
                plt.pause(0.01)
        averages.append(sum(losslist)/len(losslist))
        print(averages)
    return assignments, patterns_numpy