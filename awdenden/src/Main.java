import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.classifiers.Evaluation;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Kamal Nadjieb
 */
public class Main {
    public static void main(String args[]) throws Exception {
        /*Test test = new Test();
        test.simpleWekaTrain("C:\\Users\\Kamal Nadjieb\\Documents\\NetBeansProjects\\awdenden\\awdenden\\src\\iris.arff");
        */
        /*LOAD DATA TRAIN*/
        FileReader train_reader = new FileReader("C:\\Program Files\\Weka-3-8\\data\\iris.arff");
        Instances data_train = new Instances(train_reader);
        if(data_train.classIndex() == -1) {
            data_train.setClassIndex(data_train.numAttributes() - 1);
        }
        
        /*System.out.println(data_train.toSummaryString());
        for (int i = 0; i < data_train.numAttributes(); ++i) {
            System.out.println(data_train.attribute(0).isNumeric());
        }*/
        
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
        FFNN ffnn = new FFNN();
        ffnn.setLearningRate(0.3);
        ffnn.setNIn(0);
        ffnn.setNOut(0);
        ffnn.setNHidden(0);
        //ffnn.buildClassifier(data_train_new);
        
        /* EVALUATION */
        /*Evaluation eval = new Evaluation(data_train_new);
        eval.evaluateModel(ffnn, data_train_new);
        System.out.println(eval.errorRate()); //Printing Training Mean root squared Error
        System.out.println(eval.toSummaryString()); //Summary of Training
        */
        
        //////////////////////
        /*Instances datapredict = new Instances(
        new BufferedReader(new FileReader("C:\\Program Files\\Weka-3-8\\data\\iris.arff")));
        datapredict.setClassIndex(datapredict.numAttributes() - 1);
        Instances predicteddata = new Instances(datapredict);
            
        //Predict Part
        for (int i = 0; i < datapredict.numInstances(); i++) {
            double clsLabel = ffnn.classifyInstance(datapredict.instance(i));
            predicteddata.instance(i).setClassValue(clsLabel);
        }
        */
        
        //Storing again in arff
        BufferedWriter writer = new BufferedWriter(new FileWriter("hasil.arff"));
        writer.write(data_train_new.toString());
        writer.newLine();
        writer.flush();
        writer.close();
    }
}
