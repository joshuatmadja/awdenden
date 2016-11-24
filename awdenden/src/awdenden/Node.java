/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package awdenden;

/**
 *
 * @author user
 */
class Node {
    private double value;
    private String name;
    private double error;
    
    public Node(){
        value = 1.0;
        name = "";
        error = 0.0;
    }
    
    public Node(double value, String name){
        this.value = value;
        this.name = name;
        error=0.0;
    }
    
    public double getValue(){
        return this.value;
    }
    
    public String getName(){
        return this.name;
    }
    
    public double getError(){
        return this.error;
    }
    
    public void setValue(double value){
        this.value = value;
    }
    
    public void setName(String name){
        this.name = name;
    }
    
    public void setError(double err){
        this.error=err;
    }
}
