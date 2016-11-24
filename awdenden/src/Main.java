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
        System.out.print("Masukkan direktori data train .arff : ");
        Scanner cin = new Scanner(System.in);
        String filepath = cin.nextLine();
        FileReader train_reader = new FileReader(filepath);
        Instances data_train = new Instances(train_reader);
        if (data_train.classIndex() == -1) {
            data_train.setClassIndex(data_train.numAttributes() - 1);
        }
        
        /*NORMALIZE ATTRIBUTE*/
        Normalize norm = new Normalize();
        norm.setInputFormat(data_train);
        Instances data_train_new = Filter.useFilter(data_train, norm);
        
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
        
        /*SAVE MODEL*/
        weka.core.SerializationHelper.write("hasil.model", ffnn);
        
        /* EVALUATION */
        Evaluation eval = new Evaluation(data_train_new);
        System.out.println("1. Full training");
        System.out.println("2. 10-fold cross validation");
        System.out.println("3. Split-test");
        System.out.print("Pilih jenis evaluasi (1/2/3): ");
        int pilihan = cin.nextInt();
        switch (pilihan) {
            case 1:
                eval.evaluateModel(ffnn, data_train_new);
                break;
            case 2:
                eval.crossValidateModel(ffnn, data_train_new, 10, new Random(1));
                break;
            case 3:
                data_train_new.randomize(new Random(0));
                int trainSize = (int) Math.round(data_train_new.numInstances() * 0.8);
                int testSize = data_train_new.numInstances() - trainSize;
                Instances train = new Instances(data_train_new, 0, trainSize);
                Instances test = new Instances(data_train_new, trainSize, testSize);
                eval.evaluateModel(ffnn, test);
                break;
            default:
                break;
        }
        System.out.println(eval.toSummaryString()); //Summary of Training
        System.out.println(eval.toMatrixString()); //Confusion Matrix
        
        /*CLASSIFY DATAPREDICT*/
        System.out.print("Masukkan direktori data uji .arff : ");
        cin = new Scanner(System.in);
        filepath = cin.nextLine();
        
        Instances datapredict = new Instances(
        new BufferedReader(new FileReader(filepath)));
        datapredict.setClassIndex(datapredict.numAttributes() - 1);
        Instances predicteddata = new Instances(datapredict);
        
        /*NORMALIZE ATTRIBUTE*/
        norm.setInputFormat(datapredict);
        datapredict = Filter.useFilter(datapredict, norm);
        
        //Predict Part
        for (int i = 0; i < datapredict.numInstances(); i++) {
            double clsLabel = ffnn.classifyInstance(datapredict.instance(i));
            predicteddata.instance(i).setClassValue(clsLabel);
        }
        
        //Storing again in arff
        BufferedWriter writer = new BufferedWriter(new FileWriter("hasil.arff"));
        writer.write(predicteddata.toString());
        writer.newLine();
        writer.flush();
        writer.close();
    }
}
