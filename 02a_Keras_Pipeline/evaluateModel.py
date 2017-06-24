def evaluateModel(model, paths_test, label_test, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    
    proton_test = paths_test[~label_test][0]
    gamma_test = paths_test[label_test][0]

    paths = np.array([proton_test, gamma_test])
    labels = np.array([False, True])
    
    print('Creating prediction')
    X_test, y_test = next(batchGenerator(paths, labels, batch_size=2000, pg_ratio=0.8))
    
    
    prediction = model.predict(X_test)
    #prediction = model.predict_generator(batchGenerator(a, b, batch_size=2000, pg_ratio=0.8), steps=1)
    
    
    prediction_2 = [1 if x[0] < 0.5 else 0 for x in prediction]
    y_test_2 = [1 if x[0] < 0.5 else 0 for x in y_test]
    
    cnf_matrix = confusion_matrix(y_test_2, prediction_2)
    
    plt.figure(figsize=(6,6))
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()