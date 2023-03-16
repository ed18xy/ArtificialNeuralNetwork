package assign2;

import java.util.ArrayList;

public class Neuron {
    double output;
    ArrayList<Double> outcomingWeights;
    ArrayList<Double> input;
    Double bias;
    int weightMatrixSize;

    public Neuron(int weightMatrixSize, double bias){
        input = new ArrayList<Double>();
        outcomingWeights = new ArrayList<Double>();
        this.weightMatrixSize = weightMatrixSize;
        this.bias = bias;
    }//Neuron

    //Initialize weights randomly within provided range
    public void initializeWeights(double weightsmin, double weightsmax){
        double weight;
        outcomingWeights.clear();
        for (int i = 0; i < weightMatrixSize; i++) {
            do{
                weight = FFNN.rnd.nextDouble() * (weightsmax - weightsmin) + weightsmin;
                
            }while(weight == 0);
            outcomingWeights.add(i, weight);
        }
    }//initializeWeights

    //Calculates output of the neuron based on inputs and activation function
    public void calculateOutput(){
        double z = 0;
        for (Double i : input) {
            z+=i;
        }
        output = compute(z+bias);
        input.clear();
    }//calculateOutput

    public void addInputConnectionValue(double in){
        input.add(in);
    }//addInputConnectionValue

    public void clearInputConnections(){
        input.clear();
    }//clearInputConnections

    //Special case for input layer
    public void assignOutput(double value){
        output = value;
    }//assignOutput

    //for training
    public void updateWeightJ(int j, double w){
        outcomingWeights.set(j, outcomingWeights.get(j)-w);
    }//updateWeightJ

    //for training
    public void updateBias(double b){
        this.bias += b;
    }//updateBias

    //Sigmoid activation function
    public double compute(double in){
        return 1 / (1 + Math.exp(-in));
    }

}
