import re

rows = open("F://dataset//ibug_300W_large_face_landmark_dataset//labels_ibug_300W_train.xml").read().strip().split("\n")
output = open("F://dataset//ibug_300W_large_face_landmark_dataset//datasetLips.xml", "w")
pointsOfInterest = set(list(range(48, 60)))

partTag = re.compile("part name='[0-9]+'")

# loop over the rows of the data split file
for row in rows:
    # check to see if the current line has the (x, y)-coordinates for
    # the facial landmarks we are interested in
    parts = re.findall(partTag, row)
    # if there is no information related to the (x, y)-coordinates of
    # the facial landmarks, we can write the current line out to disk
    # with no further modifications
    if len(parts) == 0:
        output.write("{}\n".format(row))
    # otherwise, there is annotation information that we must process
    else:
        # parse out the name of the attribute from the row
        attr = "name='"
        i = row.find(attr)
        j = row.find("'", i + len(attr) + 1)
        name = int(row[i + len(attr):j])
        # if the facial landmark name exists within the range of our
        # indexes, write it to our output file
        if name in pointsOfInterest:
            output.write("{}\n".format(row))
# close the output file
output.close()
