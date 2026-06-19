#pragma once

#include <cstdlib>
#include <algorithm>

/*
  function partition(list, left, right, pivotIndex)
     pivotValue := list[pivotIndex]
     swap list[pivotIndex] and list[right]  // Move pivot to end
     storeIndex := left
     for i from left to right-1
         if list[i] < pivotValue
             swap list[storeIndex] and list[i]
             storeIndex := storeIndex + 1
     swap list[right] and list[storeIndex]  // Move pivot to its final place
     return storeIndex
*/

// The original partition algorithm I took this from only garanteed that the elements on
// the left were < pivotValue and the elements on the right were >= pivotValue.
// Unfortunately it didn't garantee that the pivotValue was in its sorted position.  By
// swapping out the pivot value before partitioning, and then swapping back in at the
// partition split, we can now garantee that the index returned is in its sorted position.
template<typename T, typename Comparator>
int partition(T* list, int left, int right, int pivotIndex, Comparator comp)
{
    T pivotValue = list[pivotIndex];
    std::swap(list[right], list[pivotIndex]);
    pivotIndex = right;
    left--;
    while (true)
    {
        do
        {
            left++;
        }
        while(left < right && comp(list[left], pivotValue));
        do
        {
            right--;
        }
        while(left < right && comp(pivotValue, list[right]));

        if ( left < right )
        {
            std::swap(list[left], list[right]);
        }
        else
        {
            // Put the pivotValue back in place
            std::swap(list[left], list[pivotIndex]);
            return left;
        }
    }
}

/*
   function select(list, left, right, k)
     loop
         select pivotIndex between left and right
         pivotNewIndex := partition(list, left, right, pivotIndex)
         if k = pivotNewIndex
             return list[k]
         else if k < pivotNewIndex
             right := pivotNewIndex-1
         else
             left := pivotNewIndex+1

*/

/*
  returns the kth largest value in the list.  A side effect is that
  list[left]..list[k-1] < list[k] < list[k+1]..list[right].
*/

template<typename T, typename Comparator>
T select_kth_element(T* list, int left, int right, int k, Comparator comp)
{
    while (true)
    {
        // select a value to pivot around between left and right and store the index to it.
        int pivotIndex = (left+right)/2;
        // Determine where this value ended up.
        int pivotNewIndex = partition<T, Comparator>(list, left, right, pivotIndex, comp);
        if (k == pivotNewIndex)
        {
            // We found the kth value
            //std::cout << "numIters = "<<numIters<<"\n";
            return list[k];
        }
        else if (k < pivotNewIndex)
        {
            // if instead we found the k+Nth value, remove the segment of the list
            // from pivotNewIndex onward from the search.
            right = pivotNewIndex-1;
        }
        else
        {
            // We found the k-Nth value, remove the segment of the list from
            // pivotNewIndex and below from the search.
            left = pivotNewIndex+1;
        }
    }
}
