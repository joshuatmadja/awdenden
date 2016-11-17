import java.io.BufferedReader;
import java.io.BufferedWriter;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import weka.classifiers.Evaluation;

import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author user
 */
public class FFNN {
    public void simpleWekaTrain(String filepath) {
        try {
        //Reading training arff or csv file
        FileReader trainreader = new FileReader(filepath);
        Instances train = new Instances(trainreader);
        train.setClassIndex(train.numAttributes() - 1);
        //Instance of NN
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        //Setting Parameters
        mlp.setLearningRate(0.1);
        mlp.setMomentum(0.2);
        mlp.setTrainingTime(2000);
        mlp.setHiddenLayers("3");
        mlp.buildClassifier(train);
        
        //////////////////////
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(mlp, train);
        System.out.println(eval.errorRate()); //Printing Training Mean root squared Error
        System.out.println(eval.toSummaryString()); //Summary of Training
        
        //////////////////////
        Instances datapredict = new Instances(
        new BufferedReader(
        new FileReader(filepath)));
        datapredict.setClassIndex(datapredict.numAttributes() - 1);
        Instances predicteddata = new Instances(datapredict);
        //Predict Part
        for (int i = 0; i < datapredict.numInstances(); i++) {
            double clsLabel = mlp.classifyInstance(datapredict.instance(i));
            predicteddata.instance(i).setClassValue(clsLabel);
        }
        //Storing again in arff
        BufferedWriter writer = new BufferedWriter(
        new FileWriter("hasil.arff"));
        writer.write(predicteddata.toString());
        writer.newLine();
        writer.flush();
        writer.close();
        }
        catch(Exception ex){
        ex.printStackTrace();
        }
}
    
    public static void main(String args[]) throws Exception {
        //DataSource source = new DataSource("iris.arff");
        //Instances dataset = source.getDataSet();
        
        //System.out.println(dataset.toSummaryString());
        
        //ArffSaver saver = new ArffSaver();
        //saver.setInstances(dataset);
        //saver.setFile(new File("hasil.arff"));
        //saver.writeBatch();
        FFNN test = new FFNN();
        test.simpleWekaTrain("C:\\Users\\Kamal Nadjieb\\Documents\\NetBeansProjects\\awdenden\\awdenden\\src\\iris.arff");
        
    }
}
