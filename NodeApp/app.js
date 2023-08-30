const express=require('express')
const request=require('request')
const app=express()

app.get('/',(req,res)=>{
    request.get('http://localhost:5004/get_image',(error,response,body)=>{

        if(!error && response.statusCode===200) {
            const data = JSON.parse(body);
            const encodedImage = data.encoded_image;
            const original_size = data.original_size;
            const compressed_size = data.compressed_size;
            const compression_ratio = data.ratio;
            console.log(data)
            
            // Process the received encoded image (e.g., render it on an EJS template)
            res.send(`
            <h2 style="font-family: Helvetica, Arial, Sans-serif">Compressed Image</h2>
            <img src="data:image/png;base64,${encodedImage}" alt="Received Image" style="height:200px; weight:210px;">
            <br>
            <br>
            <ul>
            <li>${original_size}</li>
            <li>${compressed_size}</li>
            <li>${compression_ratio}</li>
            </ul>
            `);
        }
        else{
            console.error(error)
            res.status(500).send('Error fetching image.')
    }
    })

})

app.listen(5001,()=>console.log('Server running on port 5001...'))