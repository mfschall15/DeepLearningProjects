# A Manual Approach to Multi-layered Neural Networks: 
This project taught me about neural networks and how to fine tune them. The network in the project was made without using any deep learning libraries.

# CODE DESCRIPTION
neuralnet.py file is the main file for this assignment. By launching this file, you can automatically test the
parameters in the config.yaml file. It will automatically read the keys in that file. If you want to test a new network
(for ex. with more layer, with a different learning rate etc.) just change the respective key value in the config.yaml file.

In order for this code to work, you need to have a data folder in the same folder with neuralnet.py file. This data folder
needs to contain data files. (Just like the starter code.)

NOTE: In order to test part b of the assignment you need to uncomment the following lines in the main function of neuralnet.py
file. After doing that, when you run the .py file, it will automatically print the values and differences for part b, and will
exit the code.

    # If you want to check part-b uncomment the following lines:
    ##
    #gets data from 1 of each different category
    # part_b_idx = []
    # for i in range(10):
    #     part_b_idx.append(np.where(y[:,i] == 1)[0][0])  
    # part_b_x = x[part_b_idx]
    # part_b_y = y[part_b_idx]
    
    # #Runs Part b
    # partb(model, part_b_x, part_b_y)
    # sys.exit()
    ##
