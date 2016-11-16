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
        FFNN f = new FFNN();
        f.readInstances();
        Instances jadi = f.getFiltered(f.iris);
        //System.out.println("yey");
        System.out.println(jadi);
    }
    
}
