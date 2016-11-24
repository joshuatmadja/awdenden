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
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
//        NaiveBayes008 nb = new NaiveBayes008();
//        Awdenden aw = new Awdenden();
//        double d;
//        //Instances i = nb.readInstances();
//        Instances ins = nb.getFiltered(nb.inst);
//        Instance last = ins.firstInstance();
//        nb.buildClassifier(ins);
//        d = nb.classifyInstance(last);
//        System.out.println(ins.attribute(ins.classIndex()).value((int)d));
//        aw.saveModel(ins,"NaiveBayes008");

        FFNN f = new FFNN();
        
        Instances ins = f.getFiltered(f.inst);
        f.buildClassifier(ins);
        
        Evaluation eval = new Evaluation(ins);
        eval.evaluateModel(f, ins);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }
    
}
