import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import java.io.StringReader;
import weka.core.converters.ArffLoader.ArffReader;
import java.util.Random;

public class nb {

    public static void main(String[] args) {
        try {
            // Embedded ARFF data as a string
            String arffData = 
                "@relation weather\n" +
                "\n" +
                "@attribute outlook {sunny, overcast, rainy}\n" +
                "@attribute temperature numeric\n" +
                "@attribute humidity numeric\n" +
                "@attribute windy {TRUE, FALSE}\n" +
                "@attribute play {yes, no}\n" +
                "\n" +
                "@data\n" +
                "sunny, 85, 85, FALSE, no\n" +
                "sunny, 80, 90, TRUE, no\n" +
                "overcast, 83, 78, FALSE, yes\n" +
                "rainy, 70, 96, FALSE, yes\n" +
                "rainy, 68, 80, FALSE, yes\n" +
                "rainy, 65, 70, TRUE, no\n" +
                "overcast, 64, 65, TRUE, yes\n" +
                "sunny, 72, 95, FALSE, no\n" +
                "sunny, 69, 70, FALSE, yes\n" +
                "rainy, 75, 80, FALSE, yes\n" +
                "sunny, 75, 70, TRUE, yes\n" +
                "overcast, 72, 90, TRUE, yes\n" +
                "overcast, 81, 75, FALSE, yes\n" +
                "rainy, 71, 91, TRUE, no\n";

            // Read the data from the string
            StringReader stringReader = new StringReader(arffData);
            ArffReader arff = new ArffReader(stringReader);
            Instances data = arff.getData();

            // Set class index to the last attribute (the class we want to predict)
            data.setClassIndex(data.numAttributes() - 1);

            // Create and train Naive Bayes classifier
            Classifier naiveBayes = new NaiveBayes();
            naiveBayes.buildClassifier(data);

            // Evaluate the classifier with 10-fold cross-validation
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(naiveBayes, data, 10, new Random(1));

            // Output the classifier model
            // System.out.println(naiveBayes);

            // Output overall accuracy
            System.out.println("Accuracy: " + eval.pctCorrect() + "%");

            // Output classification report (precision, recall, F1-score)
            System.out.println(eval.toClassDetailsString());

            // Output confusion matrix
            // System.out.println(eval.toMatrixString("Confusion Matrix"));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
