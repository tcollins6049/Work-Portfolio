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
