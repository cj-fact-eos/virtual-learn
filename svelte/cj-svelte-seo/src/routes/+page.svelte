<script lang="ts">
    import { Button } from '$lib/components/ui/button/index.js'
    
    // Legacy Mode
    // let count = 0;
    
    // Runes Mode
    let count = $state(0)

    function increment(){
        count++;
    }
</script>

<div>
    {count}
</div>

<Button onclick={increment}>Please click me</Button>