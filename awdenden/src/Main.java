import weka.core.Instances;
import weka.core.converters.ArffSaver;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;

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
        //Test test = new Test();
        //test.simpleWekaTrain("C:\\Users\\Kamal Nadjieb\\Documents\\NetBeansProjects\\awdenden\\awdenden\\src\\iris.arff");
        DataSource source = new DataSource("C:\\Users\\Kamal Nadjieb\\Documents\\NetBeansProjects\\awdenden\\awdenden\\src\\iris.arff");
        Instances dataset = source.getDataSet();
        
        System.out.println(dataset.toSummaryString());
        
        ArffSaver saver = new ArffSaver();
        saver.setInstances(dataset);
        saver.setFile(new File("hasil.arff"));
        saver.writeBatch();
    }
}
