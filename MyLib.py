import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
import matplotlib.font_manager
class MyLib:

    # from lab5, C is the penalty value, h is the mesh size
    @staticmethod
    def compute_and_compare_SVM(X, y, C, h):
        svc = svm.SVC(kernel='linear', C=C).fit(X=X, y=y)
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
        poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
        lin_svc = svm.LinearSVC(C=C).fit(X, y)

        # check on performance
        print 'SVC:', svc.score(X, y)
        print 'RBF_SVC:', rbf_svc.score(X, y)
        print 'Poly_SVC:', poly_svc.score(X, y)
        print 'Linear_SVC:', lin_svc.score(X, y)

        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # title for the plots
        titles = ['SVC with linear kernel',
                  'LinearSVC (linear kernel)',
                  'SVC with RBF kernel',
                  'SVC with polynomial (degree 3) kernel']

        for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            plt.subplot(2, 2, i + 1)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
            plt.xlabel('Sepal length')
            plt.ylabel('Sepal width')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
            plt.title(titles[i])

    # from lab5, comparing the svm result for with and without weight
    @staticmethod
    def compare_weighted_SVM(X, y, C, class_weight=None):
        # fit the model and get the separating hyperplane
        clf = svm.SVC(kernel='linear', C=C)
        clf.fit(X, y)
        print 'Score without weighting:', clf.score(X, y)

        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - clf.intercept_[0] / w[1]

        # get the separating hyperplane using weighted classes
        wclf = svm.SVC(kernel='linear', class_weight=class_weight)
        wclf.fit(X, y)
        print 'Score after weighting:', wclf.score(X, y)

        ww = wclf.coef_[0]
        wa = -ww[0] / ww[1]
        wyy = wa * xx - wclf.intercept_[0] / ww[1]

        # plot separating hyperplanes and samples
        h0 = plt.plot(xx, yy, 'k-', label='no weights')
        h1 = plt.plot(xx, wyy, 'k--', label='with weights')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.legend(loc='best')

        plt.axis('tight')


    @staticmethod
    def svm_novelty_detection(xx, yy, X_train, X_test, X_outliers, gamma=0.1, plot_graph=True):
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(X_train)

        TP = []
        FP = []

        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred_outliers = clf.predict(X_outliers)

        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_test[y_pred_test == -1].size
        n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

        if plot_graph:
            # plot the line, the points, and the nearest vectors to the plane
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.title("Novelty Detection")
            plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
            a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
            plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

            b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
            b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
            c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')

            plt.axis('tight')
            plt.xlim((-5, 5))
            plt.ylim((-5, 5))

            plt.legend([a.collections[0], b1, b2, c],
                       ["learned frontier", "training observations",
                        "new regular observations", "new abnormal observations"],
                       loc="upper left",
                       prop=matplotlib.font_manager.FontProperties(size=11))

            plt.xlabel(
                "error train: %d/200 ; errors novel regular: %d/40 ; "
                "errors novel abnormal: %d/40"
                % (n_error_train, n_error_test, n_error_outliers))
