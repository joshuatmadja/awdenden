/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package awdenden;

import java.util.Random;

/**
 *
 * @author user
 */
class Weight {
    private Node kiri;
    private Node kanan;
    private double value;
    private int idx;
    
    public Weight(int idx, Node kiri, Node kanan){
        this.kiri = kiri;
        this.kanan = kanan;
        value = new Random().nextDouble()-0.5;
        this.idx = idx;
    }
    
    public void setNode(Node kiri, Node kanan){
        this.kiri = kiri;
        this.kanan = kanan;
    }
    
    public Node getNodeKiri(){
        return kiri;
    }
    
    public Node getNodeKanan(){
        return kanan;
    }
    
    public void setValue(double val){
        this.value = val;
    }
    
    public double getValue(){
        return value;
    }
    
    public int getIdx(){
        return idx;
    }
}
