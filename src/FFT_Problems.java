import java.awt.Color;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Scanner;

// Make sure to add one more value when actually running


public class FFT_Problems {
	static ArrayList<Double> class1 = new ArrayList<Double>(); 
	static ArrayList<Double> tone1 = new ArrayList<Double>();
	static ArrayList<Double> tone2 = new ArrayList<Double>();
	static Complex [] z = new Complex[512];
	static Complex [] k = new Complex[512];

	double PI = Math.PI;
	double twoPi = 2 * Math.PI;
	
	public static void main(String args[]) throws FileNotFoundException {
		FFT_Problems test = new FFT_Problems();

		test.readFile(); 

		// Even and Odd values of f and g of S 
//		double S = 50;  
//		for(int i = 0; i < z.length; i++) {
//			z[i] = new Complex(test.gValues(S, class1.get(i)),0);
//		}
//		for(int i = 0; i < k.length; i++) {
//			k[i] = new Complex(test.fValues(S, class1.get(i)),0);
//		}
		
		//double [] z = test.createList(class1);
		
		//test.calcF(class1);
		//test.lowPass();
		
		
		//test.dtmfTonesA();
		//test.rangeFinder();
		test.question7();
		
		
	

	
		
	} // Main Bracket
	

	public void readFile() throws FileNotFoundException {
		FileReader inputFile = new FileReader("Resources/tone2.txt");
		Scanner fileReader = new Scanner(inputFile);
		
		while(fileReader.hasNext()) {
			String line = fileReader.nextLine(); 
			Scanner stringReader = new Scanner(line);
			tone2.add(stringReader.nextDouble());
			}
		fileReader.close();
		}
	
	public ArrayList<Double> readFile(String in) throws FileNotFoundException {
		FileReader inputFile = new FileReader(in);
		Scanner fileReader = new Scanner(inputFile);
		ArrayList<Double> temp = new ArrayList<Double>();
		
		while(fileReader.hasNext()) {
			String line = fileReader.nextLine(); 
			Scanner stringReader = new Scanner(line);
			temp.add(stringReader.nextDouble());
			}
		fileReader.close();
		
		return temp;
	}
	
	public Complex [] calcF(ArrayList<Double> list) {
		double S = 50; 
		Complex [] z = new Complex[list.size()];
		for(int i = 0; i < z.length; i++) {
			z[i] = new Complex(fValues(S, class1.get(i)),0);
		}
		return z;
		
	}
	
	public Complex [] calcG(ArrayList<Double> list) {
		double S = 50; 
		Complex [] z = new Complex[list.size()];
		for(int i = 0; i < z.length; i++) {
			z[i] = new Complex(fValues(S, class1.get(i)),0);
		}
		return z;
	}
	
	public double fValues(double s,double t) {
        double sum = 0;
        for (int i=1; i<=s; i++)
        {
            double numerator = Math.sin(twoPi * (2 * i-1)*t);
            double answer = numerator / ((2 * i) - 1);
            sum+=answer;
        }
        return sum; 
	}
	
	public double gValues(double s, double t) {
        double sum = 0;
        for(int i=1; i<=s; i++)
        {
            sum += (Math.sin(twoPi * ((2 * i) * t)) / (2 * i));
        }
        return sum;
	}
	
	public Complex [] fft(int direction, Complex[] z, int n) {
		
		double twoPi = Math.PI * 2;
		double theta;
		Complex w; 
		Complex u; 
		Complex t; 
		int r;
		int i;
		int k;
		int d = direction;
		
		
		theta = (-twoPi*d) /n;
		r = n/2;
	
		
		for(i=1;i<n;) {
            w = new Complex(Math.cos(i * theta), Math.sin(i * theta));
			for(k=0;k<n;) {
				u = new Complex(1,0);
				for(int m = 0;m<r;m++)
				{
					t = z[k+m].minus(z[k+m+r]);
					z[k+m] = z[k+m].plus(z[k+m+r]);
					z[k+m+r] = t.times(u);
					u = u.times(w);
				}
				k = k + (2 * r);
			}
			i = 2*i;
			r = r/2;
		}
		
		for(i=0;i<n;i++){
			r = i; 
			k = 0; 
			for(int m=1;m<n;m++) {
				k = (2 * k) + (r%2);
				r = r/2;
				m = 2*m;
			}
			if(k > i) {
				t = z[i];
				z[i] = z[k];
				z[k] = t;
			}
		}
		
		
		if(d==-1) {
			z = ifft(d,z,n);
		}
		return z;
	}
	
	public Complex [] ifft(int direction,Complex [] x,int n) {
		Complex newN = new Complex(n,0);
		Complex[] y = new Complex[n];
//		for(int i = 0; i < n; i++) {
//			y[i] = x[i].conjugate();
//		}
//		
//		y=fft(1,y,n);
//		
		for(int i = 0; i < n; i++) {
			y[i] = x[i].divides(newN);
		}
		return y; 
	}
	
	public double [] PSD(Complex [] z) {
		double [] psd = new double[z.length];
		for(int i = 0; i < z.length; i++) {
			Complex temp  = z[i].times(z[i].conjugate());
			double newTemp = temp.re();
			psd[i] = newTemp;
		}
		return psd;
	}

	public void evenLimits(Complex [] z) {
		z = fft(1,z,z.length); // Runnning the even values through the fft algorithm 
		double [] evenLimits = PSD(z); // Running the complex array from the fft through the psd algorithm 
		for(int i = 0; i < evenLimits.length; i++) {
			System.out.println(evenLimits[i]);
		}
	}
	
	public void oddLimits(Complex [] k) {
		k = fft(1,k,k.length); // Runnning the odd values through the fft algorithm 
		double [] oddLimits = PSD(k); // Running the complex array from the fft through the psd algorithm 
		for(int i = 0; i < oddLimits.length; i++) {
			System.out.println(oddLimits[i]);
		}
	}
	
	public void xFunction(ArrayList<Double> z) {
		double sum = 0;
		int a = 1;
		double twoPI = Math.PI * 2;
		int f1 = 13;
		int f2 = 31;
		int c = 0; 
		Complex [] xArray = new Complex[z.size()];
		for(int i = 0; i < z.size();i++) {
			double temp1 = a * Math.sin(twoPI*f1*(z.get(i) - c));
			double temp2 = a * Math.sin(twoPI*f2*(z.get(i) - c));
			sum = temp1 + temp2;
			xArray[i] = new Complex(sum,0);
		}
		xArray = fft(1,xArray,xArray.length);
		double [] test = PSD(xArray);
		printList(test);

	}
	
	public void yFunction(ArrayList<Double>  z) {
		double sum = 0;
		int a = 1;
		double twoPI = Math.PI * 2;
		int f1 = 13;
		int f2 = 31;
		int c = 0; 
		Complex [] xArray = new Complex[z.size()];
		for(int i = 0; i < z.size();i++) {
			double temp1 = a * Math.sin(twoPI*f1*(z.get(i) - c));
			double temp2 = a * Math.sin(twoPI*f2*(z.get(i) - c));
			sum = temp1 * temp2;
			xArray[i] = new Complex(sum,0);
		}
		xArray = fft(1,xArray,xArray.length);
		double [] test = PSD(xArray);
		printList(test);
	}
	
	public Complex [] createList(double [] z) {
		Complex [] k = new Complex[z.length];
		for(int i = 0; i < z.length; i++) {
			k[i] = new Complex(z[i],0);
		}
		return k;
	}
	
	public double [] createList(ArrayList<Double> lsit) {
		double [] z = new double[lsit.size()];
		for(int i = 0; i < z.length; i++) {
			z[i] = lsit.get(i);
		}
		return z;
	}
	
	public void printList(double [] z) {
		for(int i = 0; i < z.length; i++) {
			System.out.println(z[i]);
		}
	}
	
	public void printList(Complex [] z) {
		for(int i = 0; i < z.length; i++) {
			System.out.println(z[i].re());
		}
	}
		
	public void printList(Complex [][] z) {
		for(int i = 0; i < z.length; i++) {
			for(int j = 0; j < z[0].length; j++) {
				System.out.println(z[i][j].re());
			}
		}
	}
	
	public Complex [] generateSignal(int k) {
		Complex [] cArray = new Complex[256];
		for(int i = 0; i < cArray.length; i++) {
			if(i != k) {
				cArray[i] = new Complex(0,0);
			} else {
				cArray[i] = new Complex(1,0);
			}
		}
		return cArray;
	}
	
	public void q3a() {
		// Not sure if this is working right
//		Complex [] testSignal1 = generateSignal(0);
		Complex [] testSignal2 = generateSignal(10);
//		Complex [] testSignal3 = generateSignal(200);
		
//		printList(testSignal2);

//		Complex [] signal1 = fft(1,testSignal1,testSignal1.length);
		Complex [] signal2 = fft(1,testSignal2,256);
//		Complex [] signal3 = fft(1,testSignal3,256);
		
//		double [] sig1 = PSD(signal1);
		double [] sig2 = PSD(signal2);
//		double [] sig3 = PSD(signal3);
		
//		printList(sig1);
		printList(sig2);
//		printList(sig3);

		


	}
	
	public Complex[] generateSin(double c) {
		Complex [] z = new Complex[class1.size()];
		for(int i = 0; i < class1.size(); i++) {
			z[i] = new Complex(class1.get(i),0);
		}
		for(int i = 0; i < z.length; i++) {
			double x = z[i].re();
			Complex h = new Complex(Math.sin(20 * Math.PI * (x-c)),0);
			z[i] = h;
		}
		return z;	
	}
	
	public void q3b() {
//		Complex [] h0 = generateSin(0);
//		Complex [] h1 = generateSin(0.1);
//		Complex [] h2 = generateSin(0.25);
		Complex [] h3 = generateSin(0.05);

//		h0 = fft(1,h0,h0.length);
//		h1 = fft(1,h1,h1.length);
//		h2 = fft(1,h2,h2.length);
		h3 = fft(1,h3,h3.length);
		
//		double [] h0_PSD = PSD(h0);		
//		double [] h1_PSD = PSD(h1);
//		double [] h2_PSD = PSD(h2);
		double [] h3_PSD = PSD(h3);



//		printList(h0_PSD);
//		printList(h1_PSD);
//		printList(h2_PSD);
		printList(h3_PSD);


	}
	
	public void lowPass() {
		Complex [] z = calcF(class1);
		Complex [] filterSignals = new Complex[z.length]; // size 512
		Complex [] lpFilter = new Complex[z.length];

		for(int i = 0; i < z.length; i++) {
			if(i < 15 || z.length - i < 15) {
				filterSignals[i] = new Complex(1,0);
			}
			else {
				filterSignals[i] = new Complex(0,0);
			}
		}
		z = fft(1,z,z.length);
		//printList(z);
		
		// multiplies the fft z values times the filter 1 or 0 
		for(int i = 0; i < z.length; i++) {
			lpFilter[i] = z[i].times(filterSignals[i]);
		}
				
		// Computes the inverse fft  
		Complex [] LowFFT = fft(-1,lpFilter,lpFilter.length);
		printList(LowFFT);
	}
	
	public void highPass() {
		Complex [] f50 = calcF(class1);
		Complex [] filterSignal = new Complex[f50.length];
		Complex [] highPass = new Complex[f50.length];
		
		// Adds zeros in respective areas. 
		for(int i = 0; i < f50.length; i++) {
			if(i > 14 && z.length-i>14) {
				filterSignal[i] = new Complex(1,0);
			} else {
				filterSignal[i] = new Complex(0,0);
			}
		}
		
		// Takes fft of f values 
		f50 = fft(1,f50,f50.length);
		
		// multiplying frequencies * f50fft values 
		for(int i = 0; i < filterSignal.length; i++) {
			highPass[i] = f50[i].times(filterSignal[i]);
		}
		
		// inverse fft of the highpass values
		highPass = fft(-1,highPass,highPass.length);
		printList(highPass);
	}
	
	public void notchPass() {
		Complex [] z = calcF(class1);
		Complex [] filterSignal = new Complex[z.length];
		Complex [] notchPass = new Complex[z.length];
		
		for(int i = 0; i < z.length; i++) {
			if(i>8 && i < 16 || z.length-i>8 && z.length-i < 16) {
				filterSignal[i] = new Complex(0,0);
			} else {
				filterSignal[i] = new Complex(1,0);
			}
		}
		
		//printList(filterSignal);
		
		z = fft(1,z,z.length);
		
		for(int i = 0; i < filterSignal.length; i++) {
			notchPass[i] = z[i].times(filterSignal[i]);
		}
		notchPass = fft(-1,notchPass,notchPass.length);
		printList(notchPass);
	}

	public void bandPass() {
		Complex [] z = calcF(class1);
		Complex [] filterSignal = new Complex[z.length];
		Complex [] bandPass = new Complex[z.length];
		
		for(int i = 0; i < z.length; i++) {
			if(i>8 && i < 16 || z.length-i>8 && z.length-i < 16) {
				filterSignal[i] = new Complex(1,0);
			} else {
				filterSignal[i] = new Complex(0,0);
			}
		}
		//printList(filterSignal);
		z = fft(1,z,z.length);
		
		
		for(int i = 0; i < filterSignal.length; i++) {
			bandPass[i] = z[i].times(filterSignal[i]);
		}
		
		
		bandPass = fft(-1,bandPass,bandPass.length);
		printList(bandPass);
	}
	
	public void dtmfTonesB() {
		double [] tones = createList(tone2); // reads in tone 1 values
		Complex [] fftTones = createList(tones); // makes them complex 0i 
		
		fftTones = fft(1,fftTones,fftTones.length); // runs them through fft 
		double [] psdVals = PSD(fftTones); //runs them through PSD
		//printList(psdVals);
		
		int max_b1 = max(psdVals);
		int max_b2 = 145;
		
		double alpha = (max_b1 * 44100) / psdVals.length;
		double beta = (max_b2 * 44100) / psdVals.length;
		
		System.out.println(alpha);
		System.out.println(beta);

	}

	public void dtmfTonesA() {
		double [] tones = createList(tone1); // reads in tone 1 values
		Complex [] fftTones = createList(tones); // makes them complex 0i 
		
		fftTones = fft(1,fftTones,fftTones.length); // runs them through fft 
		double [] psdVals = PSD(fftTones); //runs them through PSD
		
		
		int max_a1 = 65;
		int max_a2 = 152;
		
		
		double alpha = (max_a1 * 44100) / psdVals.length;
		double beta = (max_a2 * 44100) / psdVals.length;
		
		System.out.println(alpha);
		System.out.println(beta);

	}
	
	public int max(double [] psdVals) {
		double max = psdVals[0];
		int index = 0; 
		for(int i = 1; i < psdVals.length; i++) {
			if(psdVals[i] > max) {
				max = psdVals[i];
				index = i; 
			}
		}
		return index;
	}
	
	public double max(Complex [] list) {
		double max = list[0].re();
		int index = 0; 
		for(int i = 1; i < list.length; i++) {
			if(list[i].re() > max) {
				max = list[i].re();
				//index = i;
			}
		}
		return max;
	}
	

	
	
	
	
	
	
	
	
	
	
	
	public Complex [] fastCorr(Complex [] x, Complex [] y) {
		Complex [] returnArray = new Complex[x.length];
		for(int i = 0; i < x.length; i++) {
			Complex temp = y[i].times(x[i].conjugate());
			returnArray[i] = temp;
		}
		return returnArray;
	}
	
	public void findDistance(Complex [] fcc_final) {
	    // fcc_final is the array of inverse fft of the conjugate pulse * response fft
		
		// fcc_ norm is just my double values of fcc_final / the max of fcc_final
		double [] fcc_norm = new double[fcc_final.length];
	    double max_fcc = max(fcc_final);
	    
	    // fill fcc_norm
	    for(int i = 0; i < fcc_final.length; i++)
	        fcc_norm[i] = (fcc_final[i].re() / max_fcc);
	    
	    // d will be the index of the max 
	    int d = 0;
	    for (int i = 0; i < fcc_norm.length; i++) {
	        if (fcc_norm[i] == 1)
	            d = i + 1;	 
	    };

	    System.out.println(d);

	    double T; 
	    double r; 
	    double d_dist;


	    T = 50000;
	    T *= d;
	    T += 0.002;
	    r = 1500;
	    d_dist = r * T;
	    //d_dist /= 2;

	    System.out.println(d_dist);

	}
	
	public double [] createFilter(Complex [] u, double p) {
		double [] y = new double[1024];
		double tempSum;
		for(int k = 0; k < y.length; k++) {
			if (k > p) {
				tempSum = 0.0;
				for(int i = 0; i < p; i++) {
					tempSum += u[k - i].re();
				}
				double finalSum = (1.0/p) * tempSum;
				y[k] = finalSum;
			} 
			else {
				tempSum = 0.0;
				for(int i = 0; i < k; i++) {
					tempSum += u[k-i].re();
				}
				double finalSum = (1/p) * tempSum;
				y[k] = finalSum;
			}
		}
		return y;
	}
	
	public Complex [] smoothFilter(Complex [] og, Complex [] filter) {
		Complex [] conv_fft = fft(1,og,og.length);
		Complex [] filtered_fft = fft(1,filter,filter.length);
		
		Complex [] tempList = new Complex [filter.length];
		for(int i = 0; i < filter.length; i++) {
			Complex t = conv_fft[i].times(filtered_fft[i]);
			tempList[i] = t;
		}
		return fft(-1,tempList,tempList.length);
	}
	
	public void rangeFinder() throws FileNotFoundException {
		double [] pulseList = createList(readFile("Resources/pulseSamples.txt"));
		Complex [] pulse = createList(pulseList);
		double [] rangeList = createList(readFile("Resources/1024Signals.txt"));
		Complex [] response = createList(rangeList);
		
		
		Complex [] pulse_fft = fft(1,pulse,pulse.length);
		Complex [] response_fft = fft(1,response,response.length);
		
		Complex [] corr_List = fastCorr(pulse_fft,response_fft);
		Complex [] fcc_final = fft(-1,corr_List,corr_List.length);
		findDistance(fcc_final);
		
		
		// Part a above 
//		double[] filteredResponse = createFilter(rangeComp,6);
//		Complex [] fResponse = createList(filteredResponse);
//		Complex [] smoothFilter = smoothFilter(fastCorrelation_fft,fResponse);
//				
//		double max = max(smoothFilter);
//		Complex maxComp = new Complex(max,0);
//		Complex [] finalList = new Complex [smoothFilter.length];
//		for(int i = 0; i < smoothFilter.length; i++) {
//			finalList[i] = smoothFilter[i].divides(maxComp);
//		}
		
		
		
	
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	public Complex [][] twoD_fft(Complex[][] z) {
		Complex [][] result = new Complex[z.length][z[0].length];
		for(int i = 0; i < z.length; i++) {
			for(int j = 0; j < z[0].length;j++) {
				result[i][j] = z[i][j];
			}
		}
		
		for(int i = 0; i < z[0].length;i++) {
			Complex [] column = new Complex[z[0].length];
			for(int j = 0; j < z.length;j++) {
				column[j] = result[j][i];
			}
			column = fft(1,column,column.length);
			for(int k = 0; k < z.length;k++) {
				result[k][i] = column[k];
			}
		}
		
		for(int i = 0; i < z.length; i++) {
			result[i] = fft(1,result[i],result[i].length);
		}
		return result;
	}
	
	public Complex[][] inverseTwoD_fft(Complex[][] z) {
		Complex [][] result = new Complex[z.length][z[0].length];
		for(int i = 0; i < z.length; i++) {
			for(int j = 0; j < z[0].length;j++) {
				result[i][j] = z[i][j].conjugate();
			}
		}
		
		result = twoD_fft(result);
		
		for(int i = 0; i < z.length;i++) {
			for(int j = 0; j < z[0].length; j++) {
				result[i][j] = result[i][j].conjugate();
			}
		}
		
		for(int i = 0; i < z.length;i++) {
			for(int j = 0; j < z[0].length; j++) {
				Complex temp = new Complex(z.length*z[0].length,0);
				result[i][j] = result[i][j].divides(temp);
			}
		}
		return result;
	}
	
	public void question7() {
		Picture firstPic = new Picture(512,512);
		Picture secondPic = new Picture(512,512);
		Color [][] firstColorArray = firstPic.getColorArray();
		Color [][] secondColorArray = firstPic.getColorArray();
		Complex [][] firstComplex = new Complex[firstColorArray.length][firstColorArray[0].length];
		Complex [][] secondComplex = new Complex[secondColorArray.length][secondColorArray[0].length];


		
		
		
		
		for(int i = 0; i < firstPic.height(); i++) {
			for(int j = 0; j < firstPic.width(); j++) {
				if((i > 179 && i < 320) && (j>219 && j < 331)) {
					firstPic.set(j, i, Color.white);
				} else {
					firstPic.set(j, i, Color.black);
				}
			}
		}
				
		
		for(int i = 0; i < firstPic.height(); i++) {
			for(int j = 0; j < firstPic.width(); j++) {
				if((i > 205) && (i < 205+90) && ((j > 300 && j < 331))) {
					firstPic.set(j, i, Color.black);
				} 
			}
		}



		for(int i = 0; i < secondPic.height();i++) {
			for(int j = 0; j < secondPic.width(); j++) {
				if(i < 121 && j < 30) {
					secondPic.set(j, i, Color.white);
				}
			}
		}
		for(int i = 0; i < secondPic.height();i++) {
			for(int j = 0; j < secondPic.width(); j++) {
				if((i > 15 && i < 105) && (j > 14 && j < 31)) {
					secondPic.set(j, i, Color.black);
				} 
			}
		}
		
		
		
		
		
		
		
		
		
		
//		for(int i = 0; i < firstColorArray.length; i++) {
//			for(int j = 0; j < firstColorArray[0].length; j++) {
//				int temp = firstPic.get(i, j).getRGB();
//				firstComplex[i][j] = new Complex(temp,0);
//			}
//		}
//
//		Complex [][] test = twoD_fft(firstComplex);
//		printList(test);
		
		
		
		

		

		//printList(firstComplex);

		
		
		
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
} // Class Bracket
