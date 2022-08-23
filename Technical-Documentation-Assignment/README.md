# Technical Documentation for Molecular Mass Calculator
By Tyler Collins

---

### Introduction
This document describes how to use the molecular mass calculator. This was written in the Java programming language.
Molecular mass is the mass of a given molecule and differrent molecules in the same compound can have different masses. 
This program works by finding the molecular mass of whatever the user types in. For example, if the user typed in
H2O then the program would print 18 because H = 1 and O = 16.

---

### Description of equipment and list of materials

What will you need to run this program?

- You first need 2 files, intStack.java and MolecularMass.java
- You will also need a way to run and compile these files such as Visual Studio Code

---

### Code Example

```
public static void calculate(String str)
    {
        int c = 12;
        int h = 1;
        int o = 16;
        int paren = 0;
        int temp;
        int testing;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == 'H') {
                Stack.push(h);
            }
            else if (str.charAt(i) == 'C') {
                Stack.push(c);
            }
            else if (str.charAt(i) == 'O') {
                Stack.push(o);
            }
            else if (str.charAt(i) == '(') {
                Stack.push(paren);
            }
            
            else if (str.charAt(i) == '1') {
                testing = Stack.pop();
                testing = testing * 1;
                Stack.push(testing); 
             }
            else if (str.charAt(i) == '2') {
               testing = Stack.pop();
               testing = testing * 2;
               Stack.push(testing); 
            }
            else if (str.charAt(i) == '3') {
                testing = Stack.pop();
                testing = testing * 3;
                Stack.push(testing); 
             }
             else if (str.charAt(i) == '4') {
                testing = Stack.pop();
                testing = testing * 4;
                Stack.push(testing); 
             }
             else if (str.charAt(i) == '5') {
                testing = Stack.pop();
                testing = testing * 5;
                Stack.push(testing); 
             }
             else if (str.charAt(i) == '6') {
                testing = Stack.pop();
                testing = testing * 6;
                Stack.push(testing); 
             }
             else if (str.charAt(i) == '7') {
                testing = Stack.pop();
                testing = testing * 7;
                Stack.push(testing); 
             }
             else if (str.charAt(i) == '8') {
                testing = Stack.pop();
                testing = testing * 8;
                Stack.push(testing); 
             }
             else if (str.charAt(i) == '9') {
                testing = Stack.pop();
                testing = testing * 9;
                Stack.push(testing); 
             }


            else if (str.charAt(i) == ')') {
                temp = Stack.pop();
                num = temp;
                while (temp != 0) {
                    temp = Stack.pop();
                    num += temp;
                }
                Stack.push(num);
            }

        }
        temp = Stack.pop();
        num = 0;
        while (temp != -1) {
            num += temp;
            temp = Stack.pop();
        }
        
    }

```

---

### Installation Instructions

The two files are placed in the section below, just go to whatever program you plan on using to run these files. Then make 2 new files in that directory and copy and paste each of these files to it. 

---

### How to run the program

1. First you will need to go into the program you will be using to run these files.
2. Make a new file, IntStack.java
3. Copy and paste the code below into that file or download the IntStack.java file from github.

```
public class IntStack {
	// May create private data here.
	public int [] stack;  // an array to represent the stack
	public int capacity;  // max allocated memory
	public int size;  // number of elements in array/stack


	public IntStack() {
		// TODO: Code to initialize your stack.
		size = 0;
		capacity = 100;
		stack = new int [capacity];
	}

	public void push(int x) {
		// TODO: Code to push an item x onto the stack. The stack wlil never contain more than 100 elements.
		
		stack[size] = x;
		size++;
	}

	public int pop() {
		// TODO: Code to pop and retrun an item from the top of the stack. If the stack is empty, return -1.
		int popped = -1;
		if (size == 0) {
			return popped;
		}
		else {
			int i = size - 1;
			popped = stack[i];
			stack[i] = stack[i + 1];
			size--;
			return popped;
		}
	}
}

```

4. Now make another file, MolecularMass.java
5. Copy and paste the code below into that file or download the file MolecularMass.java from github.

```
import java.util.Scanner;
public class MolecularMass {
    private static int num;
    private static IntStack Stack;
    public static void main(String[] args) {
        Stack = new IntStack();
        Scanner myObj = new Scanner(System.in);
        System.out.print("Enter the molecule: ");
        String input = myObj.nextLine();
        calculate(input);
        System.out.println("The Molecular Mass of " +  input  + " is " + num);
    } 
    public static void calculate(String str)
    {
        int c = 12;
        int h = 1;
        int o = 16;
        int paren = 0;
        int temp;
        int testing;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == 'H') {
                Stack.push(h);
            }
            else if (str.charAt(i) == 'C') {
                Stack.push(c);
            }
            else if (str.charAt(i) == 'O') {
                Stack.push(o);
            }
            else if (str.charAt(i) == '(') {
                Stack.push(paren);
            }
            
            else if (str.charAt(i) == '1') {
                testing = Stack.pop();
                testing = testing * 1;
                Stack.push(testing); 
             }
            else if (str.charAt(i) == '2') {
               testing = Stack.pop();
               testing = testing * 2;
               Stack.push(testing); 
            }
            else if (str.charAt(i) == '3') {
                testing = Stack.pop();
                testing = testing * 3;
                Stack.push(testing); 
             }
             else if (str.charAt(i) == '4') {
                testing = Stack.pop();
                testing = testing * 4;
                Stack.push(testing); 
             }
             else if (str.charAt(i) == '5') {
                testing = Stack.pop();
                testing = testing * 5;
                Stack.push(testing); 
             }
             else if (str.charAt(i) == '6') {
                testing = Stack.pop();
                testing = testing * 6;
                Stack.push(testing); 
             }
             else if (str.charAt(i) == '7') {
                testing = Stack.pop();
                testing = testing * 7;
                Stack.push(testing); 
             }
             else if (str.charAt(i) == '8') {
                testing = Stack.pop();
                testing = testing * 8;
                Stack.push(testing); 
             }
             else if (str.charAt(i) == '9') {
                testing = Stack.pop();
                testing = testing * 9;
                Stack.push(testing); 
             }


            else if (str.charAt(i) == ')') {
                temp = Stack.pop();
                num = temp;
                while (temp != 0) {
                    temp = Stack.pop();
                    num += temp;
                }
                Stack.push(num);
            }

        }
        temp = Stack.pop();
        num = 0;
        while (temp != -1) {
            num += temp;
            temp = Stack.pop();
        }
        
    }

}

```

6. Now go to the console and compile the files by either typing javac *.java, which compiles both, or
   by typing javac filename. java, which compiles the files one at a time.
7. Now run the program by typing, java MolecularMass.
8. Now type in your molecule in all uppercase letters. Feel free to use numbers and parentheses as you need.
9. Press enter and it will tell you the molecular mass.

---

### FAQs

Question: What elements does this program calculate the molecular mass for?
Answer: This program only covers hydrogen(H), oxygen(O), and carbon(C).

Question: Can you input lower case characters in this program?
Answer: No, when entering the element only use uppercase characters and numbers with no spaces.

---

### Troubleshooting/ Where to get support
Make sure you first read through this entiree document and then if you are still having problems then you can contact the writer at collinstm@appstate.edu.

---

### How to Contribute

If you have any recommendations for imroving the code or if you write some improvements of your own then email me at collinstm@appstate.edu

---

### Licensing

none

---
