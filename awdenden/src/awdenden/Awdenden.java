/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package awdenden;

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
        Instances i = nb.readInstances();
        Instances ins = nb.getFiltered(i);
        nb.buildClassifier(ins);
    }
    
}
