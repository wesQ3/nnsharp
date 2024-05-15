function convertNetwork (nn) {
   var newWeights = [];
   for (let i = 0; i < nn.Sizes.length -1; i++) {
      console.log(i, nn.Sizes[i], nn.Sizes[i+1]);
      newWeights[i] = convertTransition(nn.Weights[i], nn.Sizes[i], nn.Sizes[i+1]);
   }
   console.log(newWeights);
   return {
      weights: newWeights
   };
}

function convertTransition (array, size, count) {
   const result = [];
   for (let i = 0; i < array.length; i += size) {
      result.push(array.slice(i, i + size));
   }
   return result;
}
