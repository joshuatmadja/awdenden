import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.Random;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.classifiers.Evaluation;

public class Main {
    public static void main(String args[]) throws Exception {
        /*LOAD DATA TRAIN*/
        FileReader train_reader = new FileReader("C:\\Program Files\\Weka-3-8\\data\\iris.arff");
        Instances data_train = new Instances(train_reader);
        if(data_train.classIndex() == -1) {
            data_train.setClassIndex(data_train.numAttributes() - 1);
        }
        
        //System.out.println(data_train.instance(0).classValue());
        //System.out.println(data_train.numInstances());
        
        /*DISCRETIZE ATTRIBUTE*/
        //set options
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "first-last";
        //Apply discretization
        NominalToBinary norm = new NominalToBinary();
        norm.setOptions(options);
        norm.setInputFormat(data_train);
        Instances data_train_new = Filter.useFilter(data_train, norm);
        
        /*BUILD A NEURAL CLASSIFIER*/
        int n_in = data_train_new.numAttributes()- 1;
        int n_out = data_train_new.numClasses();
        if (n_out <= 2) {
            --n_out;
        }
        
        FFNN ffnn = new FFNN();
        ffnn.setLearningRate(0.5);
        ffnn.setNIn(n_in);
        ffnn.setNOut(n_out);
        ffnn.setNHidden(4);
        ffnn.buildClassifier(data_train_new);
        
        /* EVALUATION */
        ///*
        Evaluation eval = new Evaluation(data_train_new);
        //eval.evaluateModel(ffnn, data_train_new);
        eval.crossValidateModel(ffnn, data_train_new, 10, new Random(1));
        System.out.println(eval.errorRate()); //Printing Training Mean root squared Error
        System.out.println(eval.toSummaryString()); //Summary of Training
        //*/
        
        /*CLASSIFY DATAPREDICT*/
        ///*
        Instances datapredict = new Instances(
        new BufferedReader(new FileReader("C:\\Program Files\\Weka-3-8\\data\\iris.arff")));
        datapredict.setClassIndex(datapredict.numAttributes() - 1);
        Instances predicteddata = new Instances(datapredict);
            
        //Predict Part
        for (int i = 0; i < datapredict.numInstances(); i++) {
            double clsLabel = ffnn.classifyInstance(datapredict.instance(i));
            predicteddata.instance(i).setClassValue(clsLabel);
        }
        //*/
        
        //Storing again in arff
        BufferedWriter writer = new BufferedWriter(new FileWriter("hasil.arff"));
        writer.write(predicteddata.toString());
        writer.newLine();
        writer.flush();
        writer.close();
    }
}
