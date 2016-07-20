def plot_DT(clf,feature_names,target_names):
    '''http://scikit-learn.org/stable/modules/tree.html#tree'''
    from IPython.display import Image
    from sklearn.externals.six import StringIO
    from sklearn import tree
    import pydotplus 
    # Turn interactive plotting off
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=feature_names,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
     
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
    i=Image(graph.create_png())
    from IPython.display import display
    display(i)
    graph.write_pdf("iris.pdf") 
