package assign2;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Locale;
import java.util.Random;

public class BackProp {

    int input_size, output_size, hidden_size, max_epoch, seed, hidden_num, epoch;
    double learning_rate, hidden_weights_update, input_weights_update, hidden_bias_update, input_bias_update,
            total_error, errorO, old_total_error, total_error_test, momentum;
    int num_samples, num_samples_train, num_samples_test;
    ArrayList<ArrayList<Double>> input_vectors, hidden_error_layers;
    ArrayList<Integer> output_class, good_examples, bad_examples;
    ArrayList<Integer> train_indexes, train_indexes2, train_indexes1, test_indexes, hidden_sizes;
    ArrayList<Double> output_error_vector, hidden_error_vector;
    static Random rnd;
    FFNN network;
    ArrayList<Double> input_weights_update_hist, hidden_weights_update_hist;


    public BackProp(String config_file, String data_file, String output_file) {
        // Set up global variables
        output_class = new ArrayList<>();
        input_vectors = new ArrayList<>();
        good_examples = new ArrayList<>();
        bad_examples = new ArrayList<>();
        input_weights_update_hist = new ArrayList<>();
        hidden_weights_update_hist = new ArrayList<>();
        DecimalFormatSymbols symbols = new DecimalFormatSymbols(new Locale("en", "US"));
        symbols.setDecimalSeparator(',');
        DecimalFormat df = new DecimalFormat("#.#", symbols);
        df.setMaximumFractionDigits(6);
        total_error = 0;
        // Read parameters from configuration file
        readParameters(config_file);
        rnd = new Random(seed);
        // Get trainning and testing data
        readData(data_file);
        // Initialize network
        double[] weights_init_range = { -0.5, 0.5 };
        network = new FFNN(hidden_num, input_size, output_size, hidden_sizes, rnd.nextDouble(), weights_init_range, seed);
        // Ditribute data in train/test
        arrange_train_test_data();
        // Create output file for writing errors
        File file = new File(output_file);
        try (FileWriter writer = new FileWriter(file)){
            // Start training
            writer.write("Training1  \tTraining2  \tTest \n");
            Collections.shuffle(train_indexes);
            for (epoch = 1; epoch <= max_epoch; epoch++) {
                // Training on set 1
                train(train_indexes1);
                // Print error in training
                writer.write(df.format(total_error) + " \t");
                // Training on set 2
                train(train_indexes2);
                // Print error in training
                writer.write(df.format(total_error) + " \t");
                // Testing
                test(test_indexes);
                // Print error in testing
                writer.write(df.format(total_error_test) + " \n");
                System.out.println(epoch);
            }
        } catch (Exception e) {
            System.out.println(e);
        }

    }//BackProp

    //Tests netwrok on different data (unseen in training)
    private void test(ArrayList<Integer> testing_examples) {
        total_error_test = 0;
        for (Integer i : testing_examples) {
            network.forwardPass(input_vectors.get(i));
            double output_class_ffnn = network.outputLayer.neurons.get(0).output;
            errorO = output_class_ffnn * (1 - output_class_ffnn) * (output_class_ffnn - output_class.get(i));
            total_error_test += Math.pow(output_class.get(i) - output_class_ffnn, 2);
            if(epoch == max_epoch){
                double classification = 0;
                if(output_class_ffnn>0.5)classification  = 1;
                else classification = 0;
                
                System.out.print(output_class.get(i) +" & "+ classification);
                if(output_class.get(i) == classification)System.out.println(" & yes");
                else System.out.println(" & no");
            }
        }
    }//test

    //Trains network using back-propagation algorithm over training data set
    private void train(ArrayList<Integer> training_examples) {
        double hidden_weights_update, hidden_weights_update_old, input_weights_update, input_weights_update_old, prevError, hidden_error, z;
        ArrayList<Double> errorV;
        hidden_error = 0;
        shuffleIndexes(training_examples);
        // For each training data
        if (total_error != 0)
            old_total_error = total_error;
        total_error = 0;
        for (Integer i : training_examples) {
            // Calculate neural-net-output
            network.forwardPass(input_vectors.get(i));
            // Calculate error
            output_error_vector = new ArrayList<>();
            hidden_error_layers = new ArrayList<>();
            // Calculate error at output layer
            // !assumption for output layer to be of size 1
            double output_class_ffnn = network.outputLayer.neurons.get(0).output;
            errorO = output_class_ffnn * (1 - output_class_ffnn) * (output_class_ffnn - output_class.get(i));
            output_error_vector.add(errorO);
            total_error += Math.pow(output_class.get(i) - output_class_ffnn, 2);
            hidden_error_layers.add(output_error_vector);
            // Calculate error at hidden layer
            for(int l = hidden_num-1; l >= 0; l--) {
                hidden_error_vector = new ArrayList<>();
                if(l == hidden_num-1)prevError = errorO;
                else prevError = hidden_error;
                for (int x = 0; x < hidden_sizes.get(l); x++) {
                    double hidden_class_ffnn = network.hiddenLayers.get(l).neurons.get(x).output;
                    z = 0;
                    if(l == hidden_num-1)z = network.hiddenLayers.get(l).neurons.get(x).outcomingWeights.get(0);
                    else{
                        for(int ii = 0; ii < network.hiddenLayers.get(l+1).neurons.size(); ii++){
                            z+=network.hiddenLayers.get(l).neurons.get(x).outcomingWeights.get(ii);
                        }
                    }
                    hidden_error = hidden_class_ffnn * (1 - hidden_class_ffnn) * z * prevError;
                    hidden_error_vector.add(hidden_error);
                }
                hidden_error_layers.add(hidden_error_vector);
            }
            int count = 0;//for forward count
            // Calculate weight updates for hidden layer
            for(int l = hidden_num-1; l >= 0; l--) {
                // Assign original value to arbirtarly large number
                hidden_weights_update = 100;
                hidden_weights_update_old = 100;
                errorV = hidden_error_layers.get(count);
                for (int indexI = 0; indexI < network.hiddenLayers.get(l).neurons.size(); indexI++) {
                    for (int indexJ = 0; indexJ < errorV.size(); indexJ++) {
                        if (hidden_weights_update != 100)
                            hidden_weights_update_old = hidden_weights_update;
                        hidden_weights_update = learning_rate
                                * network.hiddenLayers.get(l).neurons.get(indexI).output
                                * errorV.get(indexJ);
                            if(!hidden_weights_update_hist.isEmpty())hidden_weights_update+=momentum * hidden_weights_update_hist.remove(0);
                            hidden_weights_update_hist.add(hidden_weights_update);
                        // update weights
                        network.hiddenLayers.get(l).neurons.get(indexI).updateWeightJ(indexJ, hidden_weights_update);
                    }
                }
                count++;
            }
            // Calculate bias update
            for (int indexJ = 0; indexJ < output_error_vector.size(); indexJ++) {
                double bias_update = learning_rate * output_error_vector.get(indexJ);
                network.outputLayer.neurons.get(indexJ).updateBias(bias_update);
            }
            // Calculate weights updates for input layer
            input_weights_update = 100;
            input_weights_update_old = 100;
            for (int indexI = 0; indexI < input_size; indexI++) {
                for (int indexJ = 0; indexJ < hidden_error_layers.get(count).size(); indexJ++) {
                    if (input_weights_update != 100)
                        input_weights_update_old = input_weights_update;
                    input_weights_update = learning_rate
                            * network.inputLayer.neurons.get(indexI).output
                            * hidden_error_layers.get(count).get(indexJ);
                            if(!input_weights_update_hist.isEmpty())input_weights_update+=momentum * input_weights_update_hist.remove(0);
                            input_weights_update_hist.add(input_weights_update);                    // update weights
                    network.inputLayer.neurons.get(indexI).updateWeightJ(indexJ, input_weights_update);
                }
            }
            int count2 = 1;//for forward count on biases
            // Calculate bias update
            for(int l = hidden_num-1; l >= 0; l--) {
                for (int indexJ = 0; indexJ < network.hiddenLayers.get(l).neurons.size(); indexJ++) {
                    double bias_update = learning_rate * hidden_error_layers.get(count2).get(indexJ);
                    network.hiddenLayers.get(l).neurons.get(indexJ).updateBias(bias_update);
                }
                count2++;
            }
            
        }
        // Adjust dynamic learning rate
        if (old_total_error >= total_error) {
            learning_rate *= 0.8;
            momentum -= 0.2;
            // keep in bounds
            if (momentum < 0)
                momentum = 0;
            if (learning_rate <= 0.05)
                learning_rate = 0.05;
        } else {
            learning_rate *= 1.2;
            momentum += 0.2;
            // keep in bounds
            if (momentum > 1)
                momentum = 1;
            if (learning_rate >= 1)
                learning_rate = 0.99;
        }
    }//train

    //Enables shuffling of data at the rate of 20%
    public void shuffleIndexes(ArrayList<Integer> list) {
        for (int i = 0; i < list.size() * 0.2; i++) {
            int index1 = rnd.nextInt(list.size() - 1);
            int index2 = rnd.nextInt(list.size() - 1);
            int trainI1 = list.get(index1);
            int trainI2 = list.get(index2);
            list.set(index2, trainI1);
            list.set(index1, trainI2);
        }
    }//shuffleIndexes

    //Reads configuration parameters for network and back-prop
    public void readParameters(String paramF) {
        hidden_sizes = new ArrayList<>();
        try {
            File pf = new File(paramF);
            Scanner scan = new Scanner(pf);
            seed = scan.nextInt();
            scan.nextLine();
            max_epoch = scan.nextInt();
            scan.nextLine();
            output_size = scan.nextInt();
            scan.nextLine();
            learning_rate = scan.nextDouble();
            scan.nextLine();
            momentum = scan.nextDouble();
            scan.nextLine();
            hidden_num = scan.nextInt();
            scan.nextLine();
            for(int i = 0; i < hidden_num; i++) {
                hidden_size = scan.nextInt();
                hidden_sizes.add(hidden_size);
                scan.nextLine();
            }
            scan.close();
        } catch (Exception e) {
            System.out.println("Error reading configuration file: " + "\n");
            e.printStackTrace();
        }
    }//readParameters

    //Reads data: classification bit plus input vector
    public void readData(String dataF) {
        try {
            File df = new File(dataF);
            Scanner scan = new Scanner(df);
            num_samples = scan.nextInt();
            input_size = scan.nextInt();
            scan.nextLine();
            for (int i = 0; i < num_samples; i++) {
                // !assumption for output layer to be of size 1
                int classification = scan.nextInt();
                if (classification == 0)
                    good_examples.add(i);
                else
                    bad_examples.add(i);
                output_class.add(classification);
                ArrayList<Double> inputs = new ArrayList<Double>();
                for (int j = 0; j < input_size; j++) {
                    double v = scan.nextDouble();
                    inputs.add(v);
                }
                input_vectors.add(inputs);
                scan.nextLine();
            }
            scan.close();
        } catch (Exception e) {
            System.out.println("Error reading data file: " + "\n");
            e.printStackTrace();
        }
    }//readData

    // Ditributes data examples in training and testing patterns (2 : 1)
    // Propotion of good and bad motors is kept
    private void arrange_train_test_data() {
        train_indexes = new ArrayList<>();
        train_indexes1 = new ArrayList<>();
        train_indexes2 = new ArrayList<>();
        test_indexes = new ArrayList<>();
        // Add examples for good motors
        for (Integer i : good_examples) {
            train_indexes.add(i);
            if (i % 3 == 0)
                train_indexes1.add(i);
            else if (i % 3 == 1)
                train_indexes2.add(i);
            else
                test_indexes.add(i);
        }
        // Add examples for bad motors
        for (Integer i : bad_examples) {
            train_indexes.add(i);
            if (i % 3 == 0)
                train_indexes1.add(i);
            else if (i % 3 == 1)
                train_indexes2.add(i);
            else
                test_indexes.add(i);
        }
    }//arrange_train_test_data
    //main
    public static void main(String[] args) {
        new BackProp("config.txt", "data.out","fileOut.txt");
        //Uncomment this to start multiple runs for different data and configuration files
        // for (int x = 1; x <= 5; x++) {
        //     String dataFile = "data/data"+x+".out";
        //         String configFile = "configurations/config"+x+".txt";
        //         String outputFile = "output/output_" + x +".out";
        //         new BackProp(configFile, dataFile, outputFile);
        // }
    }//main

}
