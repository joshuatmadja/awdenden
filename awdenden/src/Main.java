import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.Random;

import weka.core.Instances;
import weka.filters.Filter;
import weka.classifiers.Evaluation;
import weka.filters.unsupervised.attribute.Normalize;
import java.util.Scanner;

public class Main {
    public static void main(String args[]) throws Exception {
        /*LOAD DATA TRAIN*/
        System.out.print("Masukkan direktori menuju file .arff : ");
        Scanner cin = new Scanner(System.in);
        String filepath = cin.nextLine();
        FileReader train_reader = new FileReader(filepath);
        //FileReader train_reader = new FileReader("C:\\Program Files\\Weka-3-8\\data\\iris.arff");
        //FileReader train_reader = new FileReader("C:\\Users\\Kamal Nadjieb\\Documents\\ITB\\AKADEMIK\\V\\IF3170\\Tugas Besar 2\\mush.arff");
        //FileReader train_reader = new FileReader("C:\\Users\\Kamal Nadjieb\\Documents\\ITB\\AKADEMIK\\V\\IF3170\\Tugas Besar 2\\Team_test.arff");
        Instances data_train = new Instances(train_reader);
        if (data_train.classIndex() == -1) {
            data_train.setClassIndex(data_train.numAttributes() - 1);
        }
        
        /*NORMALIZE ATTRIBUTE*/
        Normalize norm = new Normalize();
        norm.setInputFormat(data_train);
        Instances data_train_new = Filter.useFilter(data_train, norm);
        //data_train_new.randomize(new Random(1));
        
        /*BUILD A NEURAL CLASSIFIER*/
        int n_in = data_train_new.numAttributes()- 1;
        int n_out = data_train_new.numClasses();
        if (n_out <= 2) {
            --n_out;
        }
        
        FFNN ffnn = new FFNN();
        ffnn.setLearningRate(0.05);
        ffnn.setNIn(n_in);
        ffnn.setNOut(n_out);
        System.out.print("Masukkan jumlah hidden node yang diinginkan: ");
        int n_hidden = cin.nextInt();
        ffnn.setNHidden(n_hidden);
        ffnn.buildClassifier(data_train_new);
        
        /* EVALUATION */
        ///*
        Evaluation eval = new Evaluation(data_train_new);
        eval.evaluateModel(ffnn, data_train_new);
        //eval.crossValidateModel(ffnn, data_train_new, 10, new Random(1));
        System.out.println(eval.toSummaryString()); //Summary of Training
        System.out.println(eval.toMatrixString()); //Confusion Matrix
        //*/
        
        /*CLASSIFY DATAPREDICT*/
        ///*
        Instances datapredict = new Instances(
        new BufferedReader(new FileReader(filepath)));
        //new BufferedReader(new FileReader("C:\\Program Files\\Weka-3-8\\data\\iris.arff")));
        //new BufferedReader(new FileReader("C:\\Users\\Kamal Nadjieb\\Documents\\ITB\\AKADEMIK\\V\\IF3170\\Tugas Besar 2\\mush.arff")));
        //new BufferedReader(new FileReader("C:\\Users\\Kamal Nadjieb\\Documents\\ITB\\AKADEMIK\\V\\IF3170\\Tugas Besar 2\\Team_test.arff")));
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
