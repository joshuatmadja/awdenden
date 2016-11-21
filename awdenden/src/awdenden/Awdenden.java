/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package awdenden;

import java.io.IOException;
import weka.classifiers.Evaluation;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

/**
 *
 * @author user
 */
public class Awdenden {

    public void saveModel(Instances i, String f) throws IOException, Exception{
        NaiveBayes008 nb = new NaiveBayes008();
        nb.buildClassifier(nb.getFiltered(i));
        SerializationHelper.write(f+".model", nb);
    }
	
	public void printConfusionMatrix(Instances i) throws IOException, Exception {
		NaiveBayes008 nb = new NaiveBayes008();
		nb.buildClassifier(nb.getFiltered(i));
		Evaluation eval = new Evaluation(i);
		eval.evaluateModel(nb, i);
		
		//hasil evaluasi
		System.out.println(eval.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("Correctly Classified Instances = " + eval.pctCorrect());
		System.out.println("Incorrectly Classified Instances = " + eval.pctIncorrect());
		System.out.println("Kappa statistic = " + eval.kappa());
		System.out.println("Mean absolute error = " + eval.meanAbsoluteError());
		System.out.println("Root mean squared error = " + eval.rootMeanSquaredError());
		System.out.println("Relative absolute error = " + eval.relativeAbsoluteError());
		System.out.println("Root relative absolute error = " + eval.rootRelativeSquaredError());
		System.out.println("Precision = " + eval.precision(1));
		System.out.println("Recall = " + eval.recall(1));
		System.out.println("Error Rate = " + eval.errorRate());
		
		System.out.println();
		
		//the confusion matrix
		System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
	}
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        NaiveBayes008 nb = new NaiveBayes008();
        Awdenden aw = new Awdenden();
        double d;
        //Instances i = nb.readInstances();
        Instances ins = nb.getFiltered(nb.inst);
        Instance last = ins.firstInstance();
        nb.buildClassifier(ins);
        d = nb.classifyInstance(last);
        System.out.println(ins.attribute(ins.classIndex()).value((int)d));
        aw.saveModel(ins,"NaiveBayes008");
		aw.printConfusionMatrix(ins);
    }
    
}
