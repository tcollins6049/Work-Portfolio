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
