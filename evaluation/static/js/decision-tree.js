/* 
Developed by Leah Scheide for the UW KnowledgeBase at the University of Wisconsin-Madison 
*/

$.fn.immediateText = function(text) {
    if (! text)
        return this.contents().not(this.children()).text();
    else
        this.contents()
            .filter(function(){ return this.nodeType == 3; })
            .first()
            .replaceWith(text);
};

$(document).ready(function(){

    $("li.option").each(function () {
        var question_text = $(this).siblings('li.question').text()
        $(this).attr('question_text', question_text)
        var orig_html = $(this).contents('span.text').html()
        $(this).attr('orig_html', orig_html)
    })
    $("ul.tree li ul").addClass("choices-collapsed"); // Hides subquestions at start

    $(".tree li input").click(function(event) { //Reveals subquestions for selected choice, hides choices not selected
      $(event.target).parent().addClass("choice-selected");
      var question_text = $(event.target).parent().attr('question_text')
      var orig_html = $(event.target).parent().attr('orig_html')
      $(event.target).parent().contents('span.text').html(
           '<span class="ref"><b>' + question_text + '</b></span>' + orig_html
      )
      $(event.target).nextAll().removeClass("choices-collapsed");
      $(event.target).parent().nextAll().addClass("choices-collapsed");
      $(event.target).parent().prevAll().addClass("choices-collapsed");
    });
  
  
    $("#goBack").click(function(event) { //Undoes last selection
        event.preventDefault();
       $('li.choice-selected:last').children("ul").addClass("choices-collapsed");
       $('li.choice-selected:last').nextAll().removeClass("choices-collapsed");
       $('li.choice-selected:last').prevAll().removeClass("choices-collapsed");
       $('li.choice-selected:last').find('span.ref').remove()
       $('li.choice-selected:last').removeClass("choice-selected")
           .children("input").prop("checked",false);
    });

    $("#reset").click(function(event) { //Resets to initial state
        event.preventDefault();
      $('.tree li').removeClass("choices-collapsed");
      $('.tree li').removeClass("choice-selected");
      $('.tree input').prop("checked",false);
      $('span.ref').remove()
      $("ul.tree li ul").addClass("choices-collapsed");
    });

    // Not currently using showAll function
        //$("#showAll").click(function(event) {
        //    $('*').removeClass("choices-collapsed");
        //    $('*').removeClass("choice-selected");
        //});

});