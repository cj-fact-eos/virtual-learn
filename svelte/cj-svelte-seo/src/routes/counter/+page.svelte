<script lang="ts">
    import { Button } from '$lib/components/ui/button/index.js'
    
    // Legacy Mode
    // let count = 0;
    
    // Runes Mode
    let count = $state(0)

    function Decrease(){
        if(count == 0){
            alert("Not less than 0")
        }
        else {
            count--;
        }
    }
   
</script>

<div class="container">
    <div class="shadow rounded bg-gray-900 px-6 py-2 text-2xl text-white">
        {count}
    </div>

    <br/>

    <input class="shadow rounded bg-gray-900 px-6 py-2 text-2xl text-white" type="number" bind:value={count} disabled />
    
    <Button class="shadow rounded bg-gray-900 px-6 py-2 text-2xl text-white" onclick={() => count++}>Increase</Button>
    
    <Button class="shadow rounded bg-gray-900 px-6 py-2 text-2xl text-white" onclick={() => {
        if(count == 0){
                alert("Not less than 0")
            }
            else {
                count--;
            }
    }}>Decrease</Button>
  </div> 
