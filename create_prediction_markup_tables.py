import json

fh = open('five_predictions.json')
predictions = json.load(fh)
#print(predictions)


for path in sorted(predictions):
    print("| Image:{0:s}             | Prediction         |".format(path.replace("squareimages/","")))
    print("|:------------:|:------------:|")
    #print(path.replace("squareimages/",""))
    probs = predictions[path]["probabilities"]
    labels = predictions[path]["predicted_labels"]
    for i,label in enumerate(labels):
        print("|{0:.2f}|{1:s}|".format(probs[i],label))
    print("")

      
            
    
