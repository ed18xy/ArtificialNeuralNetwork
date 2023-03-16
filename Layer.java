package assign2;

import java.util.ArrayList;

public class Layer {
    ArrayList<Neuron> neurons;
    public Layer(int size, double bias, double[] weightsRange, int nextLayerSize) {
        this.neurons = new ArrayList<Neuron>();
        for (int i = 0; i < size; i++) {
            Neuron n = new Neuron(nextLayerSize, bias);
            n.initializeWeights(weightsRange[0], weightsRange[1]);
            neurons.add(n);
        }
    }//Layer

    //Computes all outputs in a layer
    public void computeLayer(Layer prevLayer){
        // Send signals from previous layer
        for(int i = 0; i < this.neurons.size(); i++){
            for(int j = 0; j < prevLayer.neurons.size(); j++){
                Neuron temp = prevLayer.neurons.get(j);
                this.neurons.get(i).addInputConnectionValue(temp.output * temp.outcomingWeights.get(i));
            }
        }
        // Compute output of this layer
        for (Neuron n : neurons) {
            n.calculateOutput();
        }
    }//computeLayer

    // Special case for input layer 
    public void assignOutput(ArrayList<Double> inputVector){
        for(int i = 0; i < inputVector.size(); i++){
            neurons.get(i).assignOutput(inputVector.get(i));
        }
    }//assignOutput
}
