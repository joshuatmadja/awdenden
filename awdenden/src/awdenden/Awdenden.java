/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package awdenden;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author user
 */
public class Awdenden {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        NaiveBayes008 nb = new NaiveBayes008();
        double d;
        //Instances i = nb.readInstances();
        Instances ins = nb.getFiltered(nb.inst);
        Instance last = ins.firstInstance();
        nb.buildClassifier(ins);
        d = nb.classifyInstance(last);
        System.out.println(ins.attribute(ins.classIndex()).value((int)d));
    }
    
}
