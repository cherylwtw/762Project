import java.io.*;
import java.io.File;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import java.util.Arrays;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.converters.CSVLoader;
import java.util.Random;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.RandomForest;


public class CvParameterSelect {
    public static String dataSetFile;
    public static boolean contextOnly;
    public static Instances trainData;
    public static Instances testData;
//    public static String outputDirectory = "output/";
//    public static String outputFile;
    
    public static Instances loadInstances(String filename) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filename));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        
        return data;
    }

    public static void splitTrainAndTest(Instances data, double trainPercentage) throws Exception{
        int rnd = 13;
        data.randomize(new Random(rnd)); 
        int trainSize = (int)Math.round(data.numInstances() * trainPercentage);
        int testSize = data.numInstances() - trainSize; 
        trainData = new Instances(data, 0, trainSize); 
        testData = new Instances(data, trainSize, testSize); 
    }
    
    public static Classifier[] optimizeClassifiers(Classifier[] classifiers, Instances data) throws Exception{
        for(int i = 0; i < classifiers.length; i++){
            System.out.println("----- Optimizing: " + classifiers[i].getClass().toString() + " -----");
//            writer.println("----- Optimizing: " + classifiers[i].getClass().toString() + " -----");
            
            String optionStr = Arrays.toString(classifiers[i].getOptions());
            System.out.println("Options: " + optionStr);
//            writer.println("Options: " + optionStr);
            
            CVParameterSelection optimizedClassifier = new CVParameterSelection();
            optimizedClassifier.setDebug(true);
            optimizedClassifier.setNumFolds(10);
            optimizedClassifier.setClassifier(classifiers[i]);
            if (classifiers[i] instanceof ZeroR) {
                // No parameter needs to be added
            }
            else if (classifiers[i] instanceof NaiveBayes) {
                // two Naive Bayes model are put into the list with different options to compare manually
            }
            else if (classifiers[i] instanceof Logistic){
                optimizedClassifier.addCVParameter("M -1 10 12");
            }
            else if (classifiers[i] instanceof SMO){
//		optimizedClassifier.addCVParameter("C .1 10.1 10");
//                optimizedClassifier.addCVParameter("G .01 1.01 10");
            }
            else if (classifiers[i] instanceof J48){
		optimizedClassifier.addCVParameter("C 0.1 0.5 5");
		optimizedClassifier.addCVParameter("M 1 20 1");
            } 
            else if (classifiers[i] instanceof AdaBoostM1) {
                optimizedClassifier.addCVParameter("P 50 250 5");
            }
            else if (classifiers[i] instanceof RandomForest) {
                optimizedClassifier.addCVParameter("I 5 20 4");
                optimizedClassifier.addCVParameter("K 0 " + Math.max(20, data.numAttributes()-1) + " 1");
            }
     
            optimizedClassifier.buildClassifier(data); 
//            writer.println("After cross validation");
            System.out.println(optimizedClassifier.toSummaryString());
//            writer.println(optimizedClassifier.toSummaryString());
           
//            classifiers[i].setOptions(optimizedClassifier.getBestClassifierOptions());
            classifiers[i].setOptions(optimizedClassifier.getClassifier().getOptions());
            
            Evaluation eval = new Evaluation(data);
            
            eval.crossValidateModel(classifiers[i], data, 10, new Random(1));
            System.out.println("Validation summary\n" + eval.toSummaryString());
//            writer.println("Validation summary\n" + eval.toSummaryString());

	}
	return classifiers;
    }
    
    public static void SameFilePredict (String dataSetFile, boolean contextOnly) throws Exception{
        String dataset = dataSetFile.replace(".csv", "");
        dataset = dataset.replace("data/", "");
        Instances data = loadInstances(dataSetFile);
        
        if (contextOnly){
            // remove the first 9 column id, textual and categorical columns
            data.deleteAttributeAt(0);
            data.deleteAttributeAt(0);
            data.deleteAttributeAt(0);
            data.deleteAttributeAt(0);
            data.deleteAttributeAt(0);
            data.deleteAttributeAt(0);
            data.deleteAttributeAt(0);
            data.deleteAttributeAt(0);
            data.deleteAttributeAt(0);
        }
        else {
            // remove the first 2 bug id columns
            data.deleteAttributeAt(0);
            data.deleteAttributeAt(0);
            // remove other field than textual
            data.deleteAttributeAt(2);
            data.deleteAttributeAt(2);
            data.deleteAttributeAt(2);
            data.deleteAttributeAt(2);
            data.deleteAttributeAt(2);
        }
        
        splitTrainAndTest(data, 0.8);
          
//        Classifier zeroR_classifier =  new ZeroR();
//        
//        Classifier naive_bayes_classifier =  new NaiveBayes();
//        Classifier naive_bayes_classifier_k = new NaiveBayes();
//        String[] naive_bayes_option_list = new String[1];
//        naive_bayes_option_list[0] = "-K";
//        naive_bayes_classifier_k.setOptions(naive_bayes_option_list);
//        
//        Classifier logistic_classifier_neg8 =  new Logistic();
//        Classifier logistic_classifier_neg6 =  new Logistic();
//        Classifier logistic_classifier_neg4 =  new Logistic();
//        Classifier logistic_classifier_neg2 =  new Logistic();
//        String[] logistic_option_list = new String[2];
//        logistic_option_list[0] = "-R";
//        logistic_option_list[1] = "0.00000001";
//        logistic_classifier_neg8.setOptions(logistic_option_list);
//        logistic_option_list[0] = "-R";
//        logistic_option_list[1] = "0.000001";
//        logistic_classifier_neg6.setOptions(logistic_option_list);
//        logistic_option_list[0] = "-R";
//        logistic_option_list[1] = "0.0001";
//        logistic_classifier_neg4.setOptions(logistic_option_list);
//        logistic_option_list[0] = "-R";
//        logistic_option_list[1] = "0.01";
//        logistic_classifier_neg2.setOptions(logistic_option_list);

//        Classifier svm_classifier_poly =  new SMO();
//        String[] svm_option_list = new String[2];
//        svm_option_list[0] = "-K";
//        svm_option_list[1] = "weka.classifiers.functions.supportVector.PolyKernel";
//        svm_classifier_poly.setOptions(svm_option_list);
//        Classifier svm_classifier_rbf =  new SMO();
//        svm_option_list[0] = "-K";
//        svm_option_list[1] = "weka.classifiers.functions.supportVector.RBFKernel";
//        svm_classifier_rbf.setOptions(svm_option_list);
        
        Classifier c45_classifier =  new J48();

        Classifier[] classifierList = new Classifier[1];
//        classifierList[0] = zeroR_classifier;
//        classifierList[1] = naive_bayes_classifier;
//        classifierList[2] = naive_bayes_classifier_k;
//        classifierList[3] = logistic_classifier_neg8;
//        classifierList[4] = logistic_classifier_neg6;
//        classifierList[5] = logistic_classifier_neg4;
//        classifierList[6] = logistic_classifier_neg2;
//        classifierList[7] = svm_classifier_poly;
//        classifierList[8] = svm_classifier_rbf;
        classifierList[0] = c45_classifier;

        // use train set to tune parameters
        Classifier[] optimizedClassifierList = optimizeClassifiers(classifierList, trainData);
        
        // get accuracy on test set
        for(int i = 0; i < optimizedClassifierList.length; i++){
            Classifier classifier = optimizedClassifierList[i];
            System.out.println("***** Test Optimized: " + classifier.getClass().toString() + " *****");

            String classifier_options = Arrays.toString(classifier.getOptions()); 
            System.out.println("hyper parameters uesd = " + classifier_options); 
            
            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(optimizedClassifierList[i], testData);
            System.out.println(eval.toSummaryString());    
        }
    }
    
    public static void DifferentFilePredict(String trainDataSetFile, String testDataSetFile, boolean contextOnly) throws Exception {
        String dataset = trainDataSetFile.replace("_train.csv", "");
        dataset = dataset.replace("data/", "");
        
        Instances trainData = loadInstances(trainDataSetFile);
        Instances testData = loadInstances(testDataSetFile);

        if (contextOnly){
            // remove the first 9 column id, textual and categorical columns
            trainData.deleteAttributeAt(0);
            trainData.deleteAttributeAt(0);
            trainData.deleteAttributeAt(0);
            trainData.deleteAttributeAt(0);
            trainData.deleteAttributeAt(0);
            trainData.deleteAttributeAt(0);
            trainData.deleteAttributeAt(0);
            trainData.deleteAttributeAt(0);
            trainData.deleteAttributeAt(0);
        }
        else {
            // remove the first 2 bug id columns
            trainData.deleteAttributeAt(0);
            trainData.deleteAttributeAt(0);
        }
        
        Classifier c45_classifier =  new J48();

        Classifier[] classifierList = new Classifier[1];
        classifierList[0] = c45_classifier;

        // use train set to tune parameters
        Classifier[] optimizedClassifierList = optimizeClassifiers(classifierList, trainData);
        
        // get accuracy on test set
        for(int i = 0; i < optimizedClassifierList.length; i++){
            Classifier classifier = optimizedClassifierList[i];
            System.out.println("***** Test Optimized: " + classifier.getClass().toString() + " *****");

            String classifier_options = Arrays.toString(classifier.getOptions()); 
            System.out.println("hyper parameters uesd = " + classifier_options); 
            
            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(optimizedClassifierList[i], testData);
            System.out.println(eval.toSummaryString());    
        }
    }
    
    public static void main(String[] args) throws Exception {
        if (args.length == 2){
            dataSetFile = args[0];
            contextOnly = args[1].equalsIgnoreCase("C");
            // same file training
            SameFilePredict(dataSetFile, contextOnly);
        }
        else if (args.length == 3) {
            String trainDataSetFile = args[0];
            String testDataSetFile = args[1];
            contextOnly = args[2].equalsIgnoreCase("C");
            // same file training
            DifferentFilePredict(trainDataSetFile, testDataSetFile, contextOnly);
        }     
        else {
            return;
        }
    }
}
