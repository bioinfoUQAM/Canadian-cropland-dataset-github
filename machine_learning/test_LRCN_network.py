# import some additional librairies
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

# Write some code here to load the model from memory

# model = load_model()

# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability
test_gen = utils.image_batch_generator(test_directory, lb, batch_size, input_shape, n_classes, mode = "eval")

# predict new labels on the test set
predIdxs = model.predict(x=test_gen, steps=(test_count//batch_size))
predIdxs = np.argmax(predIdxs, axis=1)

# get the true labels of the test set
test_set = utils.create_image_sets(test_directory)

test_labels = []
for set_name in test_set:
    n = len(set_name)
    # if the set only has 2 images or less, ignore it
    
    if n > 2 and n % seq_len >= 0:
        for i in range(0, n % seq_len + 1):
            # get the first n (seq_len) elements in the list
            sequence =  set_name[i:i+seq_len]
            # do a check here to see if the list is good, then return it
            if utils.is_valid_sequence(sequence, seq_len):
                label = utils.get_classname(test_directory, sequence[0])
                encoded_label = utils.encode_label(lb, (label,))
                #print("Label : ", (label, encoded_label))
                test_labels.append(label)
 
test_labels = lb.transform(test_labels)


# There might be less predictions than actually present because of the division by batch size, 
# so we crop the test_labels matrix so the predictions and true labels have the same size

#print("Length of predictions: ", len(predIdxs))
#print("Length of test labels: ", len(test_labels))

difference = len(test_labels)-len(predIdxs)


# print the number of available sets per class category
for classname in labels:
    directory_name = test_directory + "/" + classname
    sequence_counts = utils.get_valid_sequence_count(utils.create_image_sets(directory_name), 3)
    print("Found %i sequences in testing directory %s" % (sequence_counts, classname))

print(" ")

# show a nicely formatted classification report
print("[INFO] evaluating network...")
print(classification_report(test_labels[:-difference], predIdxs,
	target_names=lb.classes_))

print(" ")
print("Confusion Matrix") 
print(confusion_matrix(test_labels[:-difference], predIdxs))


