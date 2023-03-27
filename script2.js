
  $(document).ready(()=>{
    $("#rest").chosen();
    $("#select").chosen();
    $("#select2").chosen();
    $("#bookt").chosen({disable_search_threshold: 3});
    
    $("#booko").chosen({disable_search_threshold: 3});
    
  });
  

const menu=document.querySelector('#menu');
burger.addEventListener('click',()=>{
    if(menu.classList.contains('hidden')){
        menu.classList.remove('hidden');
        
    }
    else{
        menu.classList.add('hidden');
    }
});


let form=document.querySelector('#f1');
bookt.addEventListener('click',()=>{
    bookterror.classList.add('hidden');
})
booko.addEventListener('click',()=>{
    bookoerror.classList.add('hidden');
})
appx.addEventListener('click',()=>{
    appxerror.classList.add('hidden');
})
cux.addEventListener('click',()=>{
    cuxerror.classList.add('hidden');
})
vote.addEventListener('click',()=>{
    voteerror.classList.add('hidden')
})

form.onsubmit=()=>{
    if(bookt.value==='-1'){
        bookterror.classList.remove('hidden');
        return false;
    }
    else{
        bookterror.classList.add('hidden');
    }
    if(booko.value==='-1'){
        bookoerror.classList.remove('hidden');
        return false;
    }
    else{
        bookoerror.classList.add('hidden');
    }
    if(vote.value===""){
        vote.value= "352";
        
    }
    else {
        if(parseInt(vote.value)>17000 || parseInt(vote.value)<0){
        voteerror.classList.remove('hidden');
        return false;
    }}
    if(appx.value===""){
        appx.value= "603";
        
    }
    else {
        if(parseInt(appx.value)>6000 || parseInt(appx.value)<40){
        appxerror.classList.remove('hidden');
        return false;
    }}
    if(cux.value===""){
        cux.value= "1218";
        
    }
    else {
        if(parseInt(cux.value)>2383 || parseInt(cux.value)<0){
        cuxerror.classList.remove('hidden');
        return false;
    }}
    if(select.value==='default'){
        selecterror.classList.remove('hidden');
        return false;
    }
    if(select2.value==='default'){
        select2error.classList.remove('hidden');
        return false;
    }
   
    
    let sb=document.querySelector('button');
    sb.addEventListener('click',()=>{
       outp.innerHTML="Rating is ______";
    })
    
}
var select = document.getElementById("select"),
                     arr = ['Cafes', 'Buffet', 'Delivery', 'Pubs and bars', 'Dine-out', 'Drinks & nightlife', 'Desserts'];
             
             for(var i = 0; i < arr.length; i++)
             {
                 var option = document.createElement("OPTION"),
                     txt = document.createTextNode(arr[i]);
                 option.appendChild(txt);
                 option.setAttribute("value",arr[i]);
                 select.insertBefore(option,select.lastChild);
             }
             var select2 = document.getElementById("select2"),
             arr2 = ['Koramangala 6th Block', 'Electronic City', 'Kammanahalli', 'Koramangala 4th Block', 'Basavanagudi', 'Sarjapur Road', 'Residency Road', 'Indiranagar', 'Frazer Town', 'MG Road', 'HSR', 'Bannerghatta Road', 'Church Street', 'Jayanagar', 'Lavelle Road', 'Malleshwaram', 'Rajajinagar', 'Brigade Road', 'Koramangala 5th Block', 'Bellandur', 'Brookefield', 'Kalyan Nagar', 'BTM', 'JP Nagar', 'Marathahalli', 'Whitefield', 'Koramangala 7th Block', 'Old Airport Road', 'New BEL Road', 'Banashankari'];
     
     for(var i = 0; i < arr2.length; i++)
     {
         var option = document.createElement("option"),
             txt = document.createTextNode(arr2[i]);
         option.appendChild(txt);
         option.setAttribute("value",arr2[i]);
         select2.insertBefore(option,select2.lastChild);
     }
    